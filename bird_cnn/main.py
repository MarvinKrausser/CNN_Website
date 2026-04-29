import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from bird_cnn import Bird_CNN, sample, trainCNN

from enum import Enum

from PIL import Image

SAVE_PATH = "./saved_models"
IMAGE_SIZE = 256

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

dataset = datasets.ImageFolder("data/train", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

model = Bird_CNN(c_in=3, c_hidden=64, c_out=7)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss()

trainCNN(model, optimizer, loss_module, train_loader, val_loader, device, 50, SAVE_PATH=SAVE_PATH, save=True)
exit()

image_path = "test1.jpg"

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

confidence, pred = sample(model=model, img=image, device=device, SAVE_PATH=SAVE_PATH)
print(f"Species: {bird_species(pred.item()).name} | Confidence: {int(confidence.item()*100)/100}")