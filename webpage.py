import json
import gradio as gr
import os
import wave
from vosk import Model, KaldiRecognizer
import nltk

from transcribe import convert_mp3_to_wav, transcribe_wav
from notes_converter import chunk_text, summarize_chunk
from qa_generator import load_qa_model, process_long_document
# import transcribe_live

# Download needed NLTK data
nltk.download('punkt')
nltk.download('stopwords')
# path to vosk model
MODEL_PATH = "vosk-model-en-us-0.22-lgraph"
QA_MODEL_PATH = "qa_generator_model"

# Load Vosk model
vosk_model = Model(MODEL_PATH)
qa_model, qa_tokenizer = load_qa_model(QA_MODEL_PATH)

def transcribe_audio(audio_file):
    wav_path = convert_mp3_to_wav(audio_file)

    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))

    results.append(json.loads(rec.FinalResult()))
    text = " ".join(r.get("text", "") for r in results)
    return text

def generate_notes(text):
    chunks = chunk_text(text)
    notes = []
    for chunk in chunks:
        notes.append(summarize_chunk(chunk))

    return "\n\n".join(f"- {note}" for note in notes)

def process_audio(audio_file):
    transcription = transcribe_audio(audio_file)
    notes = generate_notes(transcription)

    # Save notes to file for download
    notes_file = "generated_notes.txt"
    with open(notes_file, "w", encoding="utf-8") as f:
        f.write(notes)

    # Generate questions
    questions = process_long_document(notes, qa_model, qa_tokenizer, 2)
    questions_text = "\n\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))
    qa_file = "generated_questions.txt"
    with open(qa_file, "w", encoding="utf-8") as f:
        f.write(questions_text)

    return transcription, notes, notes_file, questions_text, qa_file

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("Lecture Notes & Practice Question Generator")
    gr.Markdown("Upload an audio file to get summarized notes and auto-generated review questions.")

    audio_input = gr.File(file_types=[".mp3", ".wav"], label="Upload audio file")

    with gr.Row():
        trans_output = gr.Textbox(label="Transcription")
        notes_output = gr.Textbox(label="Summarized Notes")

    with gr.Row():
        notes_file = gr.File(label="Download Notes (TXT)")
        qa_output = gr.Textbox(label="Generated Questions")

    qa_file = gr.File(label="Download Questions (TXT)")
    run_button = gr.Button("Generate Notes & Questions")

    run_button.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[trans_output, notes_output, notes_file, qa_output, qa_file]
    )

if __name__ == "__main__":
    demo.launch()

