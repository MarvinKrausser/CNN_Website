import os
import torch
from tqdm import tqdm
import torch.nn as nn


class Yolo_model(nn.Module):
    def __init__(self, c_in, c_hidden, boxes, img_size, grid, labels):
        super().__init__()

        self.grid = grid

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_hidden, kernel_size=7, padding=3),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=img_size*img_size*c_hidden, out_features=grid*grid*(boxes*5+labels))
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.model(x)
        x = x.reshape(batch, self.grid, self.grid, -1)
        return x
    

def train(model, loss_module, train_loader, val_loader, optimizer, SAVE_PATH, model_name, saving=True, device="cpu"):
    best_val = torch.finfo(torch.float32).max

    for epoch in range(200):
        ############
        # Training #
        ############
        model.train()

        count, lossCount = 0, 0.
        for images, labels in tqdm(train_loader, desc=f"Train", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images)

            loss = loss_module(prediction, labels)
            lossCount += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            count += images.size(0)

        train_loss = lossCount / count

        torch.cuda.empty_cache()

        ##############
        # Validation #
        ##############
        model.eval()

        count, lossCount = 0, 0.
        for images, labels in tqdm(val_loader, desc=f"Test", leave=False):
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)

                prediction = model(images)

                loss = loss_module(prediction, labels)
                lossCount += loss.item()


                lossCount += loss.sum().item()

                count += images.size(0)

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