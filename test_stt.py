
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GNANI_API_KEY")

if not api_key:
    raise RuntimeError("GNANI_API_KEY is missing from .env")

url = "https://api.vachana.ai/stt/v3"

with open("test_audio.wav", "rb") as audio:
    response = requests.post(
        url,
        headers={"X-API-Key-ID": api_key},
        data={
            "language_code": "ta-IN",
            "preferred_language": "ta-IN",
            "format": "transcribe"
        },
        files={"audio_file": ("test_audio.wav", audio, "audio/wav")},
        timeout=60
    )

print("Status:", response.status_code)
print("Response:", response.text)

if response.ok:
    result = response.json()
    print("\n🎙️ Recognized text:")
    transcribed_text= result.get("transcript", "")
    print(transcribed_prediction)