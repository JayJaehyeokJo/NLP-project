import sys
import nltk
from transformers import pipeline
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# Load summarization model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, chunk_size=400, overlap=100):
    """ Break text into small safe chunks (400 words is safe) """
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def extract_keywords(text, top_n=5):
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    stopwords = set(nltk.corpus.stopwords.words("english"))
    filtered = [w for w in words if w not in stopwords]
    freq = Counter(filtered)
    return [word for word, _ in freq.most_common(top_n)]

def summarize_chunk(text, min_length=50, max_length=150):
    # Extra truncate text if it's too long
    if len(text.split()) > 512:
        text = " ".join(text.split()[:512])
    result = summarizer(text, min_length=min_length, max_length=max_length, truncation=True)
    return result[0]["summary_text"]

def main():
    if len(sys.argv) != 2:
        print("Usage: python notes_converter.py input.txt")
        sys.exit(1)

    input_path = sys.argv[1]
    text = open(input_path, "r", encoding="utf-8").read()

    chunks = chunk_text(text)
    total = len(chunks)

    print(f"\nFound {total} chunks. Generating summarized notes...\n")
    for i, chunk in enumerate(chunks, 1):

        # Summarized Key Points
        summarized_text = summarize_chunk(chunk)
        print(f"\n- {summarized_text.strip()}")

if __name__ == "__main__":
    main()
