from enum import Enum
import os
import sqlite3
import time

import cv2
from dotenv import load_dotenv
import numpy as np
from pydantic import BaseModel
import torch
from torchvision import transforms
from fastapi import Depends, FastAPI, File, HTTPException, Header, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch.nn.functional as F

import threading

from yolo_model_production import convert_prediction, Yolo_model
from bird_cnn_production import Bird_CNN

BUILD_PATH = "./build_models"
IMAGE_SIZE_CNN = 64
IMAGE_SIZE_YOLO = 64

sem_ai = threading.Semaphore(1)

class bird_species(Enum):
    Common_Kingfisher = 0
    Common_Myna = 1
    House_Crow = 2
    Indian_Peacock = 3
    Indian_Pitta = 4
    Ruddy_Shelduck = 5
    Sarus_Crane = 6

transform_bird = transforms.Compose([
    transforms.Resize(IMAGE_SIZE_CNN),
    transforms.CenterCrop(IMAGE_SIZE_CNN),
    transforms.ToTensor()
])

transform_face = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_YOLO, IMAGE_SIZE_YOLO)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_bird = Bird_CNN(c_in=3, c_hidden=16, c_out=7)
full_path = os.path.join(BUILD_PATH, "bird_cnn")
model_bird.load_state_dict(torch.load(full_path, map_location=torch.device(device)))
model_bird.to(device)
model_bird.eval()

modeL_face = Yolo_model(c_in=3, boxes=1, grid=6, labels=1, c_hidden=16)
full_path = os.path.join(BUILD_PATH, "face_detection_yolo")
modeL_face.load_state_dict(torch.load(full_path, map_location=torch.device(device)))
modeL_face.to(device)
modeL_face.eval()

app = FastAPI()

origins = [
    "https://marvinkrausser.com",
    "https://api.marvinkrausser.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with sem_ai:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform_bird(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model_bird(image)
            probs = F.softmax(pred, dim=1)
            confidence, cls = torch.max(probs, dim=1)

        return {
            "class": bird_species(cls.item()).name,
            "confidence": confidence.item()
        }
    
@app.websocket("/predict_face")
async def predict_face(websocket: WebSocket):
    await websocket.accept()

    last = time.time()

    try:
        while True:
            jpg_bytes = await websocket.receive_bytes()

            if time.time() - last < 1 / 2.1:
                continue

            last = time.time()
            with sem_ai:

                np_arr = np.frombuffer(jpg_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)

                H, W, _ = frame.shape
                scale_w = W / IMAGE_SIZE_YOLO
                scale_h = H / IMAGE_SIZE_YOLO

                image = transform_face(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = modeL_face(image)
                    bboxes, _, _ = convert_prediction(
                        pred.squeeze(0),
                        image.squeeze(0),
                        threshold=0.9
                    )

                boxes_to_send = []
                for bbox in bboxes:
                    xmin = int(bbox[0] * scale_w)
                    ymin = int(bbox[1] * scale_h)
                    xmax = int(bbox[2] * scale_w)
                    ymax = int(bbox[3] * scale_h)
                    boxes_to_send.append([xmin, ymin, xmax, ymax])

                await websocket.send_json({"bboxes": boxes_to_send})

    except WebSocketDisconnect:
        print("Client disconnected")


load_dotenv("./database/.env")
API_KEY = os.getenv("API_KEY")
DATABASE = "./database/reviews.db"

def get_api_key(authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return authorization

class Review(BaseModel):
    website: str
    rating: int
    text: str
    date: str

sem_db = threading.Semaphore(1)

@app.get("/review")
def get_reviews(auth=Depends(get_api_key)):
    with sem_db:
        conn = sqlite3.connect(
            DATABASE,
            check_same_thread=False
        )
        cur = conn.cursor()

        cur.execute("""
        SELECT * FROM reviews;
        """)

        data = cur.fetchall()

        print(data)

        conn.close()
        return {"data": data}

@app.post("/review")
def post_review(review: Review):
    with sem_db:
        conn = sqlite3.connect(
            DATABASE,
            check_same_thread=False
        )
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO reviews (website, rating, text, created_at)
            VALUES (?, ?, ?, ?)
        """, (review.website, review.rating, review.text, review.date))

        conn.commit()
        conn.close()
        return {"status": "ok"}

@app.delete("/review")
def delete_review(auth=Depends(get_api_key)):
    with sem_db:
        conn = sqlite3.connect(
            DATABASE,
            check_same_thread=False
        )
        cur = conn.cursor()

        cur.execute("""
            DELETE FROM reviews;
        """)

        conn.commit()
        conn.close()
        return {"status": "ok"}