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

        img = img.permute(1, 2, 0)

        plt.figure(figsize=(3,3))
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")

    plt.show()

def visualizeImage(image):
    image = image.permute(1, 2, 0)

    plt.figure(figsize=(3,3))
    plt.imshow(image)
    plt.axis("off")

    plt.show()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxB_area = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = boxA_area + boxB_area - inter_area

    return inter_area / union if union > 0 else 0