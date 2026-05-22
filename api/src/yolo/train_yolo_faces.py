import os

from torchvision import transforms
from tqdm import tqdm
from yolo.yolo_dataset import YoloDataset
from yolo.yolo_model import train, Yolo_model
from yolo.yolo_loss import YoloLoss
from util import TransformedSubset, visualizeImage
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torch
import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import WeightedRandomSampler
from torchvision.utils import draw_bounding_boxes

def view_data(dataloader):
    for images, labels in iter(dataloader):
        for batch in range(images.shape[0]):
            image = images[batch]
            label = labels[batch]
            image_size = image.shape[1]
            grid_size = image_size // label.shape[0]

            boxes_to_draw = []
            grids_to_draw = []
            for x in range(label.shape[0]):
                for y in range(label.shape[1]):
                    grids_to_draw.append([x*grid_size, y*grid_size, (x+1)*grid_size, (y+1)*grid_size])
                    if label[x, y, 4].item() == 0:
                        continue
                    boxes_to_draw.append(label[x, y, :4])
            if len(boxes_to_draw) == 0:
                continue
            boxes_to_draw = torch.stack(boxes_to_draw)
            grids_to_draw = torch.tensor(grids_to_draw)
            image = draw_bounding_boxes(image, grids_to_draw, colors=(0, 255, 0))
            image = draw_bounding_boxes(image, boxes_to_draw, colors=(255, 0, 0))
            visualizeImage(image)


def train_yolo():
    SAVE_PATH = "./saved_models"
    IMAGE_SIZE = 64
    GRID = 9
    BATCH_SIZE = 1

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = YoloDataset(
        image_dir="data/faces/train", 
        annotation_path="data/faces/train/_annotations.coco.json",
        img_size=IMAGE_SIZE,
        transforms=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    model = Yolo_model(c_in=3, c_hidden=32, boxes=1, img_size=IMAGE_SIZE, grid=GRID, labels=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_module = YoloLoss()

    #view_data(train_loader)
    #exit()


    train(model=model, loss_module=loss_module, train_loader=train_loader, val_loader=val_loader, 
                optimizer=optimizer, SAVE_PATH=SAVE_PATH, saving=True, device=device, 
                model_name="face_detection_yolo")