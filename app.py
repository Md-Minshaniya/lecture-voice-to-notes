import os
from flask import Flask, render_template, request
import whisper
from transformers import pipeline

# Ensure FFmpeg path (extra safety)
os.environ["PATH"] += os.pathsep + r"D:\ffmpeg-8.0.1-essentials_build\bin"

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models safely
whisper_model = whisper.load_model("base", device="cpu")

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    framework="pt"
)

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    summary = ""
    flashcards = ""
    error = ""

    if request.method == "POST":
        audio = request.files.get("audio")

        if not audio or audio.filename == "":
            error = "No audio file uploaded."
            return render_template("index.html", error=error)

        audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
        audio.save(audio_path)

        try:
            # Transcribe with safe parameters
            result = whisper_model.transcribe(
                audio_path,
                fp16=False,
                temperature=0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )

            transcript = result.get("text", "").strip()

            if not transcript:
                error = "No speech detected in audio. Please upload a clear lecture audio."
                return render_template("index.html", error=error)

            with open("output/transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript)

            # Summarization (limit length for safety)
            summary_result = summarizer(
                transcript[:3000],
                max_length=150,
                min_length=60,
                do_sample=False
            )

            summary = summary_result[0]["summary_text"]

            with open("output/summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)

            # Flashcards
            sentences = [s.strip() for s in transcript.split(".") if len(s.strip()) > 20]
            flashcards_list = []

            for s in sentences[:5]:
                flashcards_list.append(f"Q: {s}?\nA: {s}")

            flashcards = "\n\n".join(flashcards_list)

            with open("output/flashcards.txt", "w", encoding="utf-8") as f:
                f.write(flashcards)

        except Exception as e:
            error = f"Audio processing failed: {str(e)}"

    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary,
        flashcards=flashcards,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)