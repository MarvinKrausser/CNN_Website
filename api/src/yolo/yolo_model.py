import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class Yolo_Conv_Block(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, kernel_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_hidden, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_out, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)
    
class SkipBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return(F.dropout(F.relu(self.conv_skip(x) + self.conv(x), inplace=True), p=0.3))


class Yolo_model(nn.Module):
    def __init__(self, c_in, boxes, grid, labels, c_hidden=16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),

            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            
            SkipBlock(c_in=c_hidden, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),

            nn.Conv2d(c_hidden*2, c_hidden*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((grid, grid)),
            nn.Conv2d(c_hidden*4, boxes*5 + labels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).permute(0, 2, 3, 1)
    

def train(model, loss_module, train_loader, val_loader, optimizer, SAVE_PATH, model_name, saving=True):
    best_val = torch.finfo(torch.float32).max
    device = next(model.parameters()).device

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

                count += images.size(0)

        val_loss = lossCount / count

        if(saving and best_val > val_loss):
                best_val = val_loss
                save_dir = os.path.join(SAVE_PATH, model_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, model_name)
                torch.save(model.state_dict(), save_path)

        print(f"epoch: {epoch+1} | train loss: {int(train_loss * 100000) / 100}k | val loss: {int(val_loss * 100000) / 100}k")
        torch.cuda.empty_cache()
    return best_val


def sample(model, img, device, SAVE_PATH, model_name="face_detection_yolo", folder="face_detection_yolo"):
    with torch.no_grad():
        full_path = os.path.join(SAVE_PATH, folder, model_name)
        state_dict = torch.load(full_path, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        img = img.to(device)
        pred = model(img)
        return pred