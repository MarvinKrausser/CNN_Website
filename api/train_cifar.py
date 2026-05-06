import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from bird_cnn import Bird_CNN, sample, trainCNN

SAVE_PATH = "./saved_models"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root="./data/cifar10",
    train=True,
    download=True,
    transform=transform
)

val_dataset = datasets.CIFAR10(
    root="./data/cifar10",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

model = Bird_CNN(c_in=3, c_hidden=16, c_out=10)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_module = nn.CrossEntropyLoss()

trainCNN(model, optimizer, loss_module, train_loader, val_loader, device, 50, SAVE_PATH=SAVE_PATH, save=False)