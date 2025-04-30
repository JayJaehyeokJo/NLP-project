import subprocess
import wave
import json
from vosk import Model, KaldiRecognizer
import shutil

def convert_mp3_to_wav(mp3_path, wav_path="temp.wav"):
    # ffmpeg: convert to 16 kHz mono WAV PCM
    subprocess.run([
        "ffmpeg", "-loglevel", "error",
        "-i", mp3_path,
        "-ar", "16000", "-ac", "1",
        wav_path
    ], check=True)

    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("FFmpeg is not installed or not found in PATH. Please install it to continue.")

    return wav_path

def transcribe_wav(wav_path, model_path="model"):
    # Load Vosk model (download one from https://alphacephei.com/vosk/models)
    model = Model(model_path)
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    # concatenate all text
    return " ".join(r.get("text", "") for r in results)

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python transcribe.py input.mp3 path/to/vosk-model")
        sys.exit(1)

    mp3_file = sys.argv[1]
    model_dir = sys.argv[2]

    wav_file = convert_mp3_to_wav(mp3_file)
    text = transcribe_wav(wav_file, model_dir)

    # Write transcription to input.txt
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\n--- Transcription written to input.txt ---\n")

if __name__ == "__main__":
    main()
