from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import vision
import base64
import os

# 환경변수: Render용
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/gcp-key.json"

app = FastAPI()
client = vision.ImageAnnotatorClient()

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/analyze")
async def analyze_image(request: ImageRequest):
    image_data = base64.b64decode(request.image_base64)
    image = vision.Image(content=image_data)

    # 라벨 감지
    label_response = client.label_detection(image=image)
    labels = [label.description for label in label_response.label_annotations]

    # 객체 감지
    object_response = client.object_localization(image=image)
    objects = [obj.name for obj in object_response.localized_object_annotations]

    # 텍스트 감지
    text_response = client.text_detection(image=image)
    texts = text_response.text_annotations
    detected_text = texts[0].description if texts else ""

    # 요약 구성
    summary = []
    if "Television" in labels or "TV" in objects:
        if "Cable" in labels or "Router" in objects:
            summary.append("TV 아래에 셋톱박스나 공유기와 케이블이 너저분하게 정리되지 않은 상태로 보입니다.")
        elif "Power Strip" in labels and "Router" in objects:
            summary.append("TV 아래에 멀티탭과 공유기가 있으며 깔끔하게 선정리가 되어 있는 모습입니다.")
        elif "Wall" in labels and not objects:
            summary.append("TV가 벽에 걸려 있고, 주변에는 아무것도 보이지 않습니다.")
        else:
            summary.append("TV가 벽에 설치되어 있고 주변 정리 상태는 확인되지 않았습니다.")
    else:
        summary.append("TV 관련 시공이 명확하게 인식되지 않았습니다.")

    return {
        "labels": labels,
        "objects": objects,
        "text": detected_text,
        "summary": " ".join(summary)
    }
