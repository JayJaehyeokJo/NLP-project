# Lecture Study Tool

This project is a software tool that helps students learn more efficiently by automatically transcribing lecture videos, generating concise study notes, and creating quizzes based on the material.

## Features

- **Voice-to-Text Transcription**  
  Transcribes video/audio content into accurate text using advanced speech recognition.
  (Model:VOSK https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip)

- **Note Generator**  
  Summarizes the transcribed text into structured, easy-to-read study notes.

- **Quiz Generator**  
  Creates questions from the notes to help users test their understanding.
  (Model: https://drive.google.com/drive/folders/1K9eEjrXaXPrnlC47GkkZUQfKaMzFBdDu?usp=drive_link)

## Tech Stack

- Python
- Vosk (for transcription)
- Hugging Face Transformers (for note and quiz generation)
- Gradio (for UI)

