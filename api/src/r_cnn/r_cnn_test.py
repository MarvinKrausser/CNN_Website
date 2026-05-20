import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from torch import tensor
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, functional
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from .r_cnn import ObjectDetectionCNN, train, eval
from .cocoDetectionDataset import CocoDetectionDataset
import random

SAVE_PATH = "./saved_models"
PADDING = 20

def get_transform():
    return ToTensor()

train_dataset = CocoDetectionDataset(
    image_dir="data/football/train", 
    annotation_path="data/football/train/_annotations.coco.json",
    transforms=get_transform()
)

val_dataset = CocoDetectionDataset(
    image_dir="data/football/valid",
    annotation_path="data/football/valid/_annotations.coco.json",
    transforms=get_transform()
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

model = ObjectDetectionCNN(c_in=3, c_hidden=32, c_out=2, layers=10)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss()

#train(model=model, loss_module=loss_module, train_loader=train_loader, val_loader=val_loader, 
#                optimizer=optimizer, SAVE_PATH=SAVE_PATH, saving=True, PADDING=40, device=device)

#exit()

image = next(iter(val_loader))[0][0]
eval(model=model, image=image, BUILD_PATH=os.path.join(SAVE_PATH, "object_detection", "object_detection"), 
     device=device, PADDING=40, minSize=5, maxSize=100, minConf=0.8)