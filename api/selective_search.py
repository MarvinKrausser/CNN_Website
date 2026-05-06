from collections import defaultdict
from enum import Enum
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms
import torch.nn.functional as F

from bird_cnn import Bird_CNN

from PIL import Image



BUILD_PATH = "./build_models"
IMAGE_SIZE = 64

class bird_species(Enum):
    Common_Kingfisher = 0
    CommonMyna = 1
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

img = cv2.imread("./testimages/two_crows.jpg")

if img is None:
    raise ValueError("Image not found or path is wrong")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)

ss.switchToSelectiveSearchFast()
rects = ss.process()

# convert to array for easy sorting
rects = np.array(rects)

# compute area
areas = rects[:, 2] * rects[:, 3]

# sort by area (descending)
idx = np.argsort(-areas)

# take top 10
top10 = rects[idx[:200]]
class_conf_sum = defaultdict(float)

for (x, y, w, h) in top10:
    #cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)
    crop = img[y:y+h, x:x+w]
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image = transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)
        probs = F.softmax(pred, dim=1)
        confidence, cls = torch.max(probs, dim=1)
        if confidence.item() < 0.7:
            continue

        class_conf_sum[cls.item()] += confidence.item()

for i in range(6):
    print(str(i) + ": " + str(class_conf_sum[i]))
