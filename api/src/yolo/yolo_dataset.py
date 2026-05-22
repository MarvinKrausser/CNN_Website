import math
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO 
import numpy as np


def is_center_in_grid_cell(x, y, img_w, img_h, S, cell_i, cell_j):
        cell_w = img_w / S
        cell_h = img_h / S

        # find which cell the center belongs to
        gt_cell_i = int(y / cell_h)
        gt_cell_j = int(x / cell_w)

        return (gt_cell_i == cell_i) and (gt_cell_j == cell_j)

def one_hot(index, num_classes):
    if index < 0 or index >= num_classes:
        raise ValueError(f"Index out of range, index: {index}, num_classes: {num_classes}")

    encoding = [0] * num_classes
    encoding[index] = 1
    return encoding

class YoloDataset(Dataset):
    def __init__(self, image_dir, annotation_path, img_size=64, grid = 9, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.img_size = img_size
        self.grid = grid

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        image = Image.open(image_path).convert("RGBA").convert("RGB")

        orig_w, orig_h = image.size

        scale_w = self.img_size / orig_w
        scale_h = self.img_size / orig_h

        # Load annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        boxes = []
        labels = []

        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmin, ymin, width, height = float(xmin), float(ymin), float(width), float(height)

            xmin = math.ceil(xmin * scale_w)
            ymin = math.ceil(ymin * scale_h)
            xmax = math.floor(xmin + max(width * scale_w, 1))
            ymax = math.floor(ymin + max(height * scale_h, 1))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'] - 1)

        ground_truth = list(zip(boxes, labels))
        num_labels = max(labels) + 1 if len(labels) > 0 else 1
        #[S, S, (x+y+w+h+c+C)]
        targets = np.zeros((self.grid, self.grid, 5 + num_labels))
        for x in range(self.grid):
            for y in range(self.grid):
                for i in range(len(ground_truth)):
                    box = ground_truth[i][0]
                    label = ground_truth[i][1]

                    if is_center_in_grid_cell(x=box[0]+box[2]/2, y=box[1]+box[3]/2, img_w=self.img_size, 
                                              img_h=self.img_size, S=self.grid, cell_i=x, cell_j=y):
                        
                        class_one_hot = one_hot(label, num_labels)
                        targets[x, y, 0] = box[0]  #x
                        targets[x, y, 1] = box[1]  #y
                        targets[x, y, 2] = box[2]  #w
                        targets[x, y, 3] = box[3]  #h
                        targets[x, y, 4] = 1       #confidence
                        for i in range(len(class_one_hot)):  #label
                            targets[x, y, i + 5] = class_one_hot[i]

                        del ground_truth[i]

                        break


        targets = torch.tensor(targets, dtype=torch.float32)

        # resize image
        image = image.resize((self.img_size, self.img_size))

        if self.transforms:
            image = self.transforms(image)

        return image, targets