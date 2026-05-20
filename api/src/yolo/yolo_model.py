import os
import cv2
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import time


class Yolo_model(nn.Module):
    def __init__(self, c_in, c_hidden, boxes, img_size, grid, labels):
        super().__init__()

        self.grid = grid

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_hidden, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=img_size*img_size*c_hidden, out_features=grid*grid*(boxes*5+labels))
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.model(x)
        x = x.reshape(batch, self.grid, self.grid, -1)
        return x
    
def create_stack(images, annotations, num_classes, device):
    data = []
    labels = []
    for i in range(len(images)):
        image = images[i]

        annotation = annotations[i]
        
        label_train = []
        for box, label in tuple(zip(annotation["boxes"], annotation["labels"])):
            box = box
            label = 0
            one_hot = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=num_classes)
            label_train.append(torch.cat((box, one_hot), dim=0))
        
        label_torch = torch.stack(label_train).reshape(-1).to(device)
        labels.append(label_torch)
        data.append(image)

    return data, labels
    

def train(model, num_classes, loss_module, train_loader, val_loader, optimizer, SAVE_PATH, model_name, grid, saving=True, device="cpu", img_size=64):
    best_val = torch.finfo(torch.float32).max

    for epoch in range(200):
        ############
        # Training #
        ############
        model.train()

        count, lossCount = 0, 0.
        for images, annotations in tqdm(train_loader, desc=f"Train", leave=False):
            data, labels = create_stack(images, annotations, num_classes, device)

            data = torch.stack(data).to(device)

            prediction = model(data)

            loss = []
            for pred, label in tuple(zip(prediction, labels)):
                loss.append(loss_module(pred = pred, target= label, labels=num_classes, cell_size=grid, img_w = img_size, img_h = img_size))
            loss = torch.stack(loss).to(device)
            loss = loss.sum()
            lossCount += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            count += data.size(0)

        train_loss = lossCount / count

        torch.cuda.empty_cache()

        ##############
        # Validation #
        ##############
        model.eval()

        count, lossCount = 0, 0.
        for images, annotations in tqdm(val_loader, desc=f"Test", leave=False):
            with torch.no_grad():
                data, labels = create_stack(images, annotations, num_classes, device)

            data = torch.stack(data).to(device)

            prediction = model(data)

            loss = []
            for pred, label in tuple(zip(prediction, labels)):
                loss.append(loss_module(pred = pred, target= label, labels=num_classes, cell_size=grid, img_w = img_size, img_h = img_size))
            loss = torch.stack(loss).to(device)


            lossCount += loss.sum().item()

            count += data.size(0)

        val_loss = lossCount / count

        if(saving and best_val > val_loss):
                best_val = val_loss
                save_dir = os.path.join(SAVE_PATH, model_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, model_name)
                torch.save(model.state_dict(), save_path)

        print(f"epoch: {epoch+1} | train loss: {int(train_loss * 1000) / 100} | val loss: {int(val_loss * 1000) / 100}")
        torch.cuda.empty_cache()
    return best_val