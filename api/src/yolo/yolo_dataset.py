import math
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO 
import numpy as np
from torchvision import transforms
import albumentations as A


def is_center_in_grid_cell(x, y, img_w, img_h, S, cell_i, cell_j):
        cell_w = img_w / S
        cell_h = img_h / S

        # find which cell the center belongs to
        gt_cell_i = min(int(x / cell_w), S - 1)
        gt_cell_j = min(int(y / cell_h), S - 1)

        return (gt_cell_i == cell_i) and (gt_cell_j == cell_j)

def one_hot(index, num_classes):
    if index < 0 or index >= num_classes:
        raise ValueError(f"Index out of range, index: {index}, num_classes: {num_classes}")

    encoding = [0] * num_classes
    encoding[index] = 1
    return encoding

def turn_grid_centered(x, y, img_w, img_h, S, cell_i, cell_j):
    cell_w = img_w / S
    cell_h = img_h / S

    cell_border_w = cell_w * cell_i
    cell_border_h = cell_h * cell_j

    return x - cell_border_w, y - cell_border_h

def flip_bbox_horizontal(box, image_width):
    x, y, w, h = box
    return (image_width - x - w, y, w, h)

class YoloDataset(Dataset):
    def __init__(self, image_dir, annotation_path, img_size=64, grid = 9, transform=False):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.img_size = img_size
        self.grid = grid
        self.num_classes = len(self.coco.cats)-1
        self.toTensor = transforms.ToTensor()

        if transform:
            self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.2),
                        A.Affine(
                            translate_percent=(-0.4, 0.4),
                            scale=(0.5, 1.5),
                            rotate=0,
                            border_mode=cv2.BORDER_REPLICATE,
                            p=0.5
                        ),
                        A.Resize(img_size, img_size)
                    ],
                    bbox_params=A.BboxParams(
                        format="coco",
                        label_fields=["labels"],
                        min_visibility=0.5
                )
            )
        else:
            self.transform = A.Compose(
                    [
                        A.Resize(img_size, img_size)
                    ],
                    bbox_params=A.BboxParams(
                        format="coco",
                        label_fields=["labels"],
                        min_visibility=0.3
                )
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        image =  cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        boxes = []
        labels = []

        annotations = sorted(annotations, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)

        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmin, ymin, width, height = int(float(xmin)), int(float(ymin)), int(float(width)), int(float(height))

            boxes.append([xmin, ymin, width, height])
            labels.append(obj['category_id'] - 1)

        augmented = self.transform(
            image=image,
            bboxes=boxes,
            labels=labels
        )

        image = augmented["image"]
        boxes = augmented["bboxes"]
        labels = augmented["labels"]

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        ground_truth = list(zip(boxes, labels))
        #[S, S, (x+y+w+h+c+C)]
        targets = np.zeros((self.grid, self.grid, 5 + self.num_classes), dtype=np.float32)
        for x in range(self.grid):
            for y in range(self.grid):
                for i in range(len(ground_truth)):
                    box = ground_truth[i][0]
                    label = ground_truth[i][1]

                    x_center = box[0] + box[2] / 2
                    y_center = box[1] + box[3] / 2

                    if is_center_in_grid_cell(x=x_center, y=y_center, img_w=self.img_size, 
                                              img_h=self.img_size, S=self.grid, cell_i=x, cell_j=y):
                        
                        class_one_hot = one_hot(int(label), self.num_classes)
                        x_grid_centered, y_grid_centered = turn_grid_centered(x=x_center, y=y_center, img_w=self.img_size, 
                                              img_h=self.img_size, S=self.grid, cell_i=x, cell_j=y)


                        targets[x, y, 0] = x_grid_centered / (self.img_size / self.grid)
                        targets[x, y, 1] = y_grid_centered / (self.img_size / self.grid)

                        targets[x, y, 2] = box[2] / self.img_size
                        targets[x, y, 3] = box[3] / self.img_size
                        targets[x, y, 4] = 1
                        for j in range(len(class_one_hot)):
                            targets[x, y, j + 5] = class_one_hot[j]

                        del ground_truth[i]
                        break


        targets = torch.tensor(targets, dtype=torch.float32)
        image = self.toTensor(image)

        return image, targets