import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from util import TransformedSubset, visualizeData
from .bird_cnn import Bird_CNN, sample, trainCNN

from enum import Enum

from PIL import Image

SAVE_PATH = "../saved_models"
IMAGE_SIZE = 128

class bird_species(Enum):
    Common_Kingfisher = 0
    Common_Myna = 1
    House_Crow = 2
    Indian_Peacock = 3
    Indian_Pitta = 4
    Ruddy_Shelduck = 5
    Sarus_Crane = 6

transform_augemnt = transforms.Compose([
    transforms.RandomAffine(
        degrees=35,                # no rotation
        translate=(0.2, 0.2)      # shift up to 20% horizontally/vertically
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/CUB_200_2011/images")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_subset, val_subset = random_split(dataset, [train_size, val_size])
train_dataset = TransformedSubset(train_subset, transform_augemnt)
val_dataset = TransformedSubset(val_subset, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

model = Bird_CNN(c_in=3, c_hidden=16, c_out=200)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss()

trainCNN(model, optimizer, loss_module, train_loader, val_loader, device, 500, SAVE_PATH=SAVE_PATH, save=True)
exit()

image_path = "test1.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

confidence, pred = sample(model=model, img=image, device=device, SAVE_PATH=SAVE_PATH)
print(f"Species: {bird_species(pred.item()).name} | Confidence: {int(confidence.item()*100)/100}")