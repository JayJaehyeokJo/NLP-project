import json
import gradio as gr
from gradio.themes.base import Base
import os
import wave
from vosk import Model, KaldiRecognizer
import nltk

from transcribe import convert_mp3_to_wav
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
    if os.path.exists("temp.wav"):
        try:
            os.remove("temp.wav")
        except Exception as e:
            print(f"Failed to delete temp audio file: {e}")
    wav_path = convert_mp3_to_wav(audio_file)

    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    results = []
    # progress(0, desc="Starting transcription...")
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
    # progress(0.5, desc="Generating notes...")
    for chunk in chunks:
        notes.append(summarize_chunk(chunk))

    # progress(1.0, desc="Notes generation complete!")
    return "\n\n".join(f"- {note}" for note in notes)


def process_audio(audio_file):
    progress = gr.Progress(track_tqdm=True)
    yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    progress(0, desc="Starting transcription...")
    transcription = transcribe_audio(audio_file)

    progress(0.5, desc="Generating notes...")
    notes = generate_notes(transcription)

    # save transcription to file for download
    transcript_file = "transcript.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcription)

    # Save notes to file for download
    notes_path = "generated_notes.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes)

    progress(1.0, desc="Finished!")

    # delete temp.wav file
    if os.path.exists(audio_file):
        try:
            os.remove(audio_file)
        except Exception as e:
            print(f"Failed to delete temp audio file: {e}")

    yield (
        gr.update(value=transcript_file, visible=True),
        gr.update(value=notes_path, visible=True),
        gr.update(interactive=True, visible=True)
    )


def generate_questions(notes):
    progress = gr.Progress(track_tqdm=True)
    yield gr.update(visible=False), gr.update(visible=False)

    with open(notes.name, "r", encoding="utf-8") as f:
        notes_text = f.read()

    # Generate questions
    progress(0, desc="Generating questions...")
    raw_questions = process_long_document(notes_text, qa_model, qa_tokenizer, 2)
    questions = []
    qa_pairs = []
    for chunk in raw_questions:
        for q in chunk.get("questions", []):
            if "[SEP]" in q:
                question, answer = q.split("[SEP]", 1)
                question = question.strip()
                answer = answer.strip()
            else:
                question = q.strip()
                answer = "N/A"
            questions.append(question)
            qa_pairs.append((question, answer))

    # combined Q/A
    combined_path = "combined_qa.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        for i, (question, answer) in enumerate(qa_pairs):
            f.write(f"Q{i+1}: {question}\n")
            f.write(f"A{i+1}: {answer}\n\n")

    # only questions
    qa_path = "generated_questions.txt"
    with open(qa_path, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions):
            f.write(f"Q{i+1}: {question}\n")

    progress(1.0, desc="Finished!")

    yield (
        gr.update(value=combined_path, visible=True),
        gr.update(value=qa_path, visible=True)
    )



# theme = Base(
  #  primary_hue="blue",
   # secondary_hue="red",
    # neutral_hue="slate",
    # font="monospace"
# )

# Gradio Interface
with gr.Blocks(theme=Base(primary_hue="blue", secondary_hue="red", neutral_hue="slate", font="monospace")) as demo:
    gr.Markdown("# <center>Lecture Notes & Practice Question Generator<center>")
    gr.Markdown("<center><p style='font-size: 12px;'>Upload an audio file to get summarized notes and auto-generated "
                "review questions.</p><center>")

    with gr.Row():
        audio_input = gr.File(file_types=[".mp3", ".wav"], label="Upload audio file", type="filepath")

    with gr.Row():
        run_button = gr.Button("Generate Transcript & Notes", scale=2)

    with gr.Row():
        with gr.Column(scale=2, min_width=300):
            transcript = gr.File(label="Download Transcript (TXT)", visible=False)
            notes_file = gr.File(label="Download Notes (TXT)", visible=False)
            questions_button = gr.Button("Generate Questions", interactive=False, visible=False)

    with gr.Row():
        questions_file = gr.File(label="Download Questions (TXT)", visible=False)
        qa_file = gr.File(label="Download Answers (TXT)", visible=False)

    run_button.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcript, notes_file, questions_button],
        show_progress="full"
    )

    questions_button.click(
        fn=generate_questions,
        inputs=[notes_file],
        outputs=[questions_file, qa_file],
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch()
