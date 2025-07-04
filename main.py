from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from agents.lie_detect_agent import LieDetectAgent
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = LieDetectAgent()

@app.post("/analyze")
async def analyze(image: UploadFile = File(None), audio: UploadFile = File(None), text: str = Form(None)):
    img = None
    audio_path = None

    # Handle image input and ensure it's in RGB format
    if image:
        img = Image.open(io.BytesIO(await image.read()))
        if img.mode != "RGB":
            img = img.convert("RGB")  # Convert RGBA or grayscale to RGB

    # Handle audio input and save it to disk
    if audio:
        os.makedirs("assets", exist_ok=True)
        audio_path = f"assets/{audio.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

    # Run analysis
    result = agent.analyze(image=img, audio_file=audio_path, text=text)
    return result
