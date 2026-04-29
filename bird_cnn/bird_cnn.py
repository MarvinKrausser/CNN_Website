import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class  SeparableConvolution(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, kernel_size, groups=c_in, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x
    
class SkipBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.conv = SeparableConvolution(c_in=c_in, c_out=c_out, kernel_size=kernel_size)
        self.conv_skip = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return(F.relu(self.conv_skip(x) + self.conv(x)))


class Bird_CNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(),

            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),

            SkipBlock(c_in=c_hidden, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),

            SeparableConvolution(c_in=c_hidden*2, c_out=c_hidden*4, kernel_size=3),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden*4, c_out),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.model(x)


def trainCNN(model, optimizer, loss_module, train_data_loader, validation_data_loader, device, num_epochs, SAVE_PATH, save=False):

    best_val = 0
    for epoch in range(num_epochs):
        ############
        # Training #
        ############
        model.train()

        true_preds, count = 0, 0
        for data_inputs, classes in tqdm(train_data_loader, desc=f"Train Epoch {epoch+1}", leave=False):
            data_inputs = data_inputs.to(device)
            classes = classes.to(device)

            preds = model(data_inputs)

            loss = loss_module(preds, classes)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            true_preds += (preds.argmax(dim=1) == classes).sum().item()
            count += data_inputs.size(0)
        train_acc = true_preds / count

        torch.cuda.empty_cache()

        ##############
        # Validation #
        ##############
        model.eval()

        true_preds, count = 0, 0
        for data_inputs, classes in tqdm(validation_data_loader, desc=f"Validate Epoch {epoch+1}", leave=False):
            with torch.no_grad():
                data_inputs = data_inputs.to(device)
                classes = classes.to(device)

                preds = model(data_inputs)

                loss = loss_module(preds, classes)

                true_preds += (preds.argmax(dim=1) == classes).sum().item()
                count += data_inputs.size(0)
        val_acc = true_preds / count

        if(save and best_val < val_acc):
            best_val = val_acc
            save_dir = os.path.join(SAVE_PATH, "bird_cnn")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"bird_cnn{epoch+1}")
            torch.save(model.state_dict(), save_path)

        print(f"epoch: {epoch+1} | train accuracy: {int(train_acc * 1000) / 10}% | validation accuracy: {int(val_acc * 1000) / 10}%")
        torch.cuda.empty_cache()


def sample(model, img, device, SAVE_PATH, model_name="bird_cnn", folder="bird_cnn"):
    with torch.no_grad():
        full_path = os.path.join(SAVE_PATH, folder, model_name)
        state_dict = torch.load(full_path, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        img = img.to(device)
        pred = model(img)
        probs = F.softmax(pred, dim=1)
        return(torch.max(probs, dim=1))