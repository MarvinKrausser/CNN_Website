from PIL import Image
import os
from matplotlib import pyplot as plt
import torch

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.subset)
    

def visualizeData(dataset):
    images, labels = next(iter(dataset))

    for i in range(4):
        img = images[i]

        # Convert tensor shape from [C,H,W] -> [H,W,C]
        img = img.permute(1, 2, 0)

        plt.figure(figsize=(3,3))
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")

    plt.show()