# app.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from google.cloud import vision
import base64
import os

# 환경변수: 키 파일 경로
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/vision-key.json"


app = FastAPI()
client = vision.ImageAnnotatorClient()

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/analyze")
async def analyze_image(request: ImageRequest):
    image_data = base64.b64decode(request.image_base64)
    image = vision.Image(content=image_data)
    response = client.label_detection(image=image)  # 예: 객체 라벨 인식

    labels = [label.description for label in response.label_annotations]
    return {"labels": labels}
