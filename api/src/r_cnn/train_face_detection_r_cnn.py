import os

from torchvision import transforms
from tqdm import tqdm
from util import TransformedSubset, visualizeImage
from api.src.r_cnn.r_cnn import ObjectDetectionCNN, trainNormalDataset, eval
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import WeightedRandomSampler



SAVE_PATH = "./saved_models"
PADDING = 20
IMAGE_SIZE = 64

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

transform_augemnt = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/faces_processed")

weights = [1/3, 1]

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(dataset),
    replacement=True
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_subset, val_subset = random_split(dataset, [train_size, val_size])
train_dataset = TransformedSubset(train_subset, transform_augemnt)
val_dataset = TransformedSubset(val_subset, transform)


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

model = ObjectDetectionCNN(c_in=3, c_hidden=32, c_out=2, layers=10)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss(weight=torch.tensor([1/3, 1], device=device, dtype=torch.float32))

#trainNormalDataset(model=model, loss_module=loss_module, train_loader=train_loader, val_loader=val_loader, 
#                optimizer=optimizer, SAVE_PATH=SAVE_PATH, saving=True, device=device, model_name="face_detection")

#exit()

image = read_image("data/faces/train/_url-http_3A_2F_2Fdingyue-ws-126-net_2F2023_2F0201_2F6a17ca80j00rpd7yx00bmd000dw00gop_jpg.rf.Setza3JikTSzG04c0Fd1.jpg")
image = image.float() / 255.0
visualizeImage(image)

eval(model=model, image=image, BUILD_PATH=os.path.join(SAVE_PATH, "face_detection", "face_detection"), device=device, minConf=0.97)