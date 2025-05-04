import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
import time
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


nltk.download('punkt', quiet=True)

def parse_args():

    parser = argparse.ArgumentParser(description="Train a Q&A generation model using SQuAD dataset")
    parser.add_argument("--squad_path", type=str, default="/kaggle/input/train-v1-1-json/train-v1.1.json", help="Path to SQuAD dataset file")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./qa_generator_model", help="Directory to save the model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args=[])

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_squad_data(squad_path):
    """Load and preprocess SQuAD dataset"""
    logger.info(f"Loading SQuAD dataset from {squad_path}")

    with open(squad_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

    examples = []
    for data in squad_data['data']:
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            if len(context.split()) < 10:
                continue
            for qa in paragraph['qas']:
                if not qa.get('is_impossible', False):
                    question = qa['question']
                    if len(qa['answers']) > 0:
                        answer = qa['answers'][0]['text']
                        examples.append({
                            'context': context,
                            'question': question,
                            'answer': answer
                        })


    logger.info(f"Extracted {len(examples)} question-answer pairs from the dataset")
    return examples

def split_text_into_chunks(text, max_words=100, overlap=20):
    """Split longer texts into manageable chunks with overlap"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_point = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_point:]
            current_length = len(current_chunk)
        current_chunk.extend(words)
        current_length += len(words)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

class QADataset(Dataset):
    """Dataset for question-answer generation"""
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = f"generate question: {example['context']}"
        target_text = f"{example['question']} [SEP] {example['answer']}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = targets.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train(args):
    """Train the model"""
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    examples = load_squad_data(args.squad_path)

    train_examples, val_examples = train_test_split(
        examples, test_size=0.1, random_state=args.seed
    )
    logger.info(f"Training on {len(train_examples)} examples, validating on {len(val_examples)} examples")
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    train_dataset = QADataset(train_examples, tokenizer, args.max_length)
    val_dataset = QADataset(val_examples, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(f"Starting training on {device}")

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Average training loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            loss = outputs.loss
            val_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Average validation loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"Saving model with improved validation loss: {avg_val_loss:.4f}")

            model_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

    logger.info("Saving final model")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return model, tokenizer

def test_model(model, tokenizer, examples, device, max_length=512):
    """Test the model on a few examples"""
    model.eval()
    model.to(device)
    test_examples = np.random.choice(examples, min(5, len(examples)), replace=False)

    for i, example in enumerate(test_examples):
        context = example['context']
        input_text = f"generate question: {context}"
        input_ids = tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        try:
            question, answer = output_text.split("[SEP]")
            question = question.strip()
            answer = answer.strip()
        except:
            question = output_text
            answer = "N/A"

        logger.info(f"\nExample {i+1}:")
        logger.info(f"Context: {context[:100]}...")
        logger.info(f"Actual Question: {example['question']}")
        logger.info(f"Generated Question: {question}")
        logger.info(f"Actual Answer: {example['answer']}")
        logger.info(f"Generated Answer: {answer}")

class QuizGenerator:
    """Class for generating questions from text"""
    def __init__(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate_questions(self, text, num_questions=5, max_length=512):
        """Generate questions from a text passage"""
        chunks = split_text_into_chunks(text)
        all_questions = []
        for chunk in chunks:
            input_text = f"generate question: {chunk}"
            input_ids = self.tokenizer(
                input_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=100,
                    num_beams=5,
                    num_beam_groups=5,
                    num_return_sequences=min(3, num_questions),
                    temperature=0.8,
                    diversity_penalty=0.5,
                    early_stopping=True
                )
            for output in outputs:
                output_text = self.tokenizer.decode(output, skip_special_tokens=True)
                try:
                    question, answer = output_text.split("[SEP]")
                    question = question.strip()
                    answer = answer.strip()
                    if len(question) < 10 or len(answer) < 5:
                        continue
                    all_questions.append({
                        "question": question,
                        "answer": answer,
                        "context": chunk
                    })
                except:
                    continue
            if len(all_questions) >= num_questions:
                break
        return all_questions[:num_questions]

def main():
    args = parse_args()
    model, tokenizer = train(args)
    examples = load_squad_data(args.squad_path)

    #test run
    #examples = examples[:20]

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model(model, tokenizer, examples, device, args.max_length)
    logger.info(f"Model saved to {args.output_dir}")
    #Example
    generator = QuizGenerator(args.output_dir)
    example_text = """
    Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics.
    """
    questions = generator.generate_questions(example_text, num_questions=3)
    logger.info("\nExample Quiz Generation:")
    logger.info(f"Text: {example_text}")
    for i, q in enumerate(questions):
        logger.info(f"\nQuestion {i+1}: {q['question']}")
        logger.info(f"Answer: {q['answer']}")

if __name__ == "__main__":

    main()

