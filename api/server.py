from enum import Enum
import os

import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch.nn.functional as F

from bird_cnn import Bird_CNN

import threading

BUILD_PATH = "./build_models"
IMAGE_SIZE = 64

sem = threading.Semaphore(1) #adjust to performance

class bird_species(Enum):
    Common_Kingfisher = 0
    Common_Myna = 1
    House_Crow = 2
    Indian_Peacock = 3
    Indian_Pitta = 4
    Ruddy_Shelduck = 5
    Sarus_Crane = 6

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Bird_CNN(c_in=3, c_hidden=16, c_out=7)
full_path = os.path.join(BUILD_PATH, "bird_cnn")
model.load_state_dict(torch.load(full_path, map_location=torch.device(device)))
model.to(device)
model.eval()

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
    with sem:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image)
            probs = F.softmax(pred, dim=1)
            confidence, cls = torch.max(probs, dim=1)

        return {
            "class": bird_species(cls.item()).name,
            "confidence": confidence.item()
        }