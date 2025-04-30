from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from nltk.tokenize import sent_tokenize
import nltk

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# path = r"C:\Users\Rohith CHekuri\Documents\utd_mscs_spring25\NLP\project\qa_generator_model"
# nltk.data.path.append(path)

def load_qa_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer


def split_text_into_chunks(text):  # , max_words=100, overlap=20):
    # sentences = sent_tokenize(text)
    # chunks = []
    # current_chunk = []
    # current_length = 0

    # for sentence in sentences:
    #     words = sentence.split()

    #     if current_length + len(words) > max_words and current_chunk:
    #         chunks.append(' '.join(current_chunk))
    #         overlap_point = max(0, len(current_chunk) - overlap)
    #         current_chunk = current_chunk[overlap_point:]
    #         current_length = len(current_chunk)

    #     current_chunk.extend(words)
    #     current_length += len(words)

    chunks = text.split("-")
    # if current_chunk:
    #     chunks.append(' '.join(current_chunk))

    return chunks


def generate_questions_from_text(model, tokenizer, text, num_questions=2):
    """Generate questions from a given text chunk"""
    input_text = f"Context: {text}\nGenerate questions:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=num_questions
        )

    questions = []
    for output in output_sequences:
        question = tokenizer.decode(output, skip_special_tokens=True)
        questions.append(question)

    return questions


def process_long_document(document_text, model, tokenizer, questions_per_chunk=2):  # chunk_size=100, chunk_overlap=20,
    chunks = split_text_into_chunks(document_text)  # , max_words=chunk_size, overlap=chunk_overlap)

    all_questions = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}:")
        print(f"Chunk text (first 100 chars): {chunk[:100]}...")

        questions = generate_questions_from_text(model, tokenizer, chunk, num_questions=questions_per_chunk)

        chunk_questions = {
            "chunk_index": i,
            "chunk_text": chunk,
            "questions": questions
        }

        all_questions.append(chunk_questions)

        print(f"Generated questions for chunk {i + 1}:")
        for j, q in enumerate(questions):
            print(f"  Question {j + 1}: {q}")

    return all_questions


if __name__ == "__main__":
    model_path = r"C:\Users\Rohith CHekuri\Documents\utd_mscs_spring25\NLP\project\qa_generator_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    long_text = """
    - This course will explore the complex interdependent systems that regulate the earth's environment as we examine exactly how human activity is influencing them. We'll investigate and define the key issues in the global warming debate as professor mcelroy guides us through intricate scientific concepts. This rigorous inquiry will lead us to a comprehensive understanding of the science behind theglobal warming crisis.
    - Professor mcelroy guides us through intricate scientific concepts as well as explains the reaction of the world community. This rigorous inquiry will lead us to a comprehensive understanding of the science behind the global warming crisis. You will certainly learn a great deal by listening to professor mcilroy's lectures you'll get the most out of this course if you pursue the additional studies outlined in the accompanying course guide here.
    - Global warming is a very complicated issue it has complex scientific dimensions it involves how the atmosphere and the ocean combined as a system. We need to understand it today in order to decide what we should do about it now a few years ago a famous scientist called roger revelle described this issue in rather graphic terms he said we are embarked on a global geophysical experiment.
    - i thought it'd be useful in this first lecture to provide a context in which to view the human presence we need to have a sense of where we are on the stage of life. i want to explicitly begin to discuss in lecture to the concept of the greenhouse effect. greenhouse effect is basically the phenomenon that maintains the surface of the earth at a temperature that is that is hospitable to life.
    - We'll discuss what that greenhouse effect is why is it called the greenhouse effect. We'll be talking about but how efficient is the natural greenhouse effect are we to understand how that works in the soil that we have a better sense of how we are beginning to change it.
    - Major gases in the atmosphere like water vapor like carbon dioxide like nitrous oxide like methane are the key players in determining the greenhouse effect and ultimately then the average temperature of the earth. After the industrial revolution and our ability to begin to use fossil fuels coal and oil and to build machines we were effectively free to change the environment in a fashion that we could not before.
    - lecture six will begin to talk about is a very important greenhouse gas that is changing namely carbon dioxide. We'll talk about how we know that and we will see at that point is that what's happening right now is effectively unprecedented the level of carbon dioxide is on a vertical climb and we are responsible for that change.
    - There's no controversy here the story here is straightforward even the most skeptical person who does not believe in environmental issues and wants- There's no controversy here the story here is straightforward even the most skeptical person who does not believe in environmental issues and wants to run away from it has to admit that the material up to this point is absolutely straightforward humans are changing the composition of the atmosphere globally. The heat engine that runs the terrestrial climate system is the tropics.
    - Climate system is not just a matter of worrying about the atmosphere also got to worry about the ocean the ocean plays a very important role in determining that kind of climate we have on the earth. We'll talk about how the ocean functions the system how does it play a role in deciding the climate that we have today.
    - Lecturer: To assess the significance of the changes in climate that are taking place today you obviously need to have a sense of history. Earth begins four and a half billion years ago it's fall formed from a spinning mass of gas and dust that represent the composition of the original solar nebula.
    - The early days of the earth really is not preserved we don't we can't go to a rock and interrogate the rock to see what was the condition under which that rock form. The oldest rocks we can find at the surface of the the earth are older significantly than the formation of the Earth.
    - There are two possibilities one is that the seeds of life were formed elsewhere in the universe perhaps even in the outer regions of the primitive solar nebula and it rays rained literally manna from heaven to provide the seeds for the eventual plethora of lifeforms that are developed on the earth. i rather think there's a more likely possibility which is that it formed in situ on the Earth.
    - How did life get to the earth well frankly scientists don't really know there there are two possibilities one is that the seeds of life were formed elsewhere in the universe perhaps even in the outer regions of the primitive solar nebula and it rays rained literally manna from heaven to provide the seeds for the eventual plethora of lifeforms that are developed on the earth.
    - The evolution of life was a relatively slow process and by evolution here i mean the the the transition to more complex lifeforms. The early lifeforms were simple unicellular as i said precarious not through time life begins to become a little bit more complicated and we'd begin to see multicellular organisms develop.
    - Life is not an evolution is not as some people believe a steady darwinian process where the strong survive and the weak die away that's not really the way we think about it. Evolution occurs almost in steps there are times of rapid environmental change which leads to rapid change in the type of organisms that can survive it.
    - The earth was smacked by and by a large meteor or by a series of meteors and these meteors triggered global climate change. The first real evidence for a massive global flash a change that had clear implications for some of our genetic ancestors.
    - The idea that humans said some real inbuilt respect for nature and were conditioned to do the right thing is probably not supported by the history of life on the planet. Over a very short period time after our human ancestors arrived they had in fact eliminated many species. The industrial revolution was when man first became liberated essentially from dependence on human and animal labor.
    - The industrial revolution was when man first became liberated essentially from dependence on human and animal labor and began to be able to use machines in a very efficient way. The issue that we're going to be dealing with in this course is what's going to happen over the next few tenths of a second of of our history.
    - In the next lecture we're actually going to digress we're going to talk about another environmental issue this is the issue of stratospheric ozone depletion. This is an example of a global issue that caught public attention on a global basis where leaders and nations came together with success in dealing with the problem.
    """

    print("Starting document processing...")
    results = process_long_document(
        long_text,
        model,
        tokenizer,
        # chunk_size=80,
        # chunk_overlap=15,
        questions_per_chunk=1
    )

    print("\nProcessing complete!")
    total_questions = sum(len(chunk["questions"]) for chunk in results)
    print(f"Generated {total_questions} questions from {len(results)} text chunks.")