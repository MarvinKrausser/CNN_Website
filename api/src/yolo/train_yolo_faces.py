import os

from torchvision import transforms
from tqdm import tqdm
from yolo_model import train, Yolo_model
from yolo_loss import YoloLoss
from cocoDetectionDataset import CocoDetectionDatasetResized
from util import TransformedSubset, visualizeImage
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import WeightedRandomSampler



SAVE_PATH = "./saved_models"
IMAGE_SIZE = 128
GRID = 9
BATCH_SIZE = 1

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CocoDetectionDatasetResized(
    image_dir="data/faces/train", 
    annotation_path="data/faces/train/_annotations.coco.json",
    img_size=IMAGE_SIZE,
    transforms=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_subset, val_subset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

model = Yolo_model(c_in=3, c_hidden=32, boxes=2, img_size=IMAGE_SIZE, grid=GRID, labels=1)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = YoloLoss()

train(model=model, loss_module=loss_module, train_loader=train_loader, val_loader=val_loader, 
            optimizer=optimizer, SAVE_PATH=SAVE_PATH, saving=True, device=device, 
            model_name="face_detection_yolo", img_size=IMAGE_SIZE, num_classes=1, grid=GRID)

exit()