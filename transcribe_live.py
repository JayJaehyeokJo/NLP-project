import sys
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = sys.argv[1]  # path to your model directory

# Load Vosk
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)
recognizer.SetWords(True)

# Create or clear input.txt at start
with open("input.txt", "w", encoding="utf-8") as f:
    f.write("")

def callback(indata, frames, time, status):
    if status:
        print(f"⚠️ {status}", file=sys.stderr)

    data_bytes = bytes(indata)

    if recognizer.AcceptWaveform(data_bytes):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        if text:
            with open("input.txt", "a", encoding="utf-8") as f:
                f.write(text + " ")
    # No need to print partial results anymore

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_live.py <model_path>")
        sys.exit(1)

    try:
        with sd.RawInputStream(
            samplerate=16000, blocksize=8000, dtype="int16",
            channels=1, callback=callback
        ):
            print("🎤 Listening (Ctrl+C to stop)…")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n🛑 Stopped. Final transcription saved in input.txt")
