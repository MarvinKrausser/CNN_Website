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


class Bird_CNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(),
            nn.Conv2d(c_hidden, c_hidden, 3, stride=2, padding=1)
        )

        # 1x1 conv branch
        self.branch1 = SeparableConvolution(c_in=c_hidden, c_out=64, kernel_size=1)

        # 1x1 -> 3x3 conv branch
        self.branch2 = SeparableConvolution(c_in=c_hidden, c_out=128, kernel_size=3)

        # 1x1 -> 5x5 conv branch
        self.branch3 = SeparableConvolution(c_in=c_hidden, c_out=32, kernel_size=5)

        # 3x3 max pooling -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_hidden, 32, kernel_size=1),
            nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, c_out)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv_init(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = F.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        return self.linear(x)


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