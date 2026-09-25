"""Cutting object and background crops out of detection images."""

import random

import torch
import torchvision.transforms.functional as TF

from src.common.boxes import iou


def random_crop(W, H, sizeX=64, sizeY=64):
    x = random.randint(0, max(W - sizeX, 0))
    y = random.randint(0, max(H - sizeY, 0))
    return x, y, x + sizeX, y + sizeY


def is_background(crop, gt_boxes, threshold=0.0):
    return all(iou(crop, box) <= threshold for box in gt_boxes)


def get_background_crop(image, gt_boxes, max_trials=100, iou_threshold=0.0):
    """A random crop that overlaps no ground-truth box, or None."""
    _, H, W = image.shape
    for _ in range(max_trials):
        x1, y1, x2, y2 = random_crop(W, H, sizeX=random.randint(50, 600), sizeY=random.randint(50, 600))
        if is_background((x1, y1, x2, y2), gt_boxes, threshold=iou_threshold):
            return image[:, y1:y2, x1:x2]
    return None


def create_stack(images, annotations, padding, crop_size=64, backgrounds_per_image=0):
    """Crops every annotated box (label from the annotation) plus
    `backgrounds_per_image` background crops (label 0) per image."""
    crops, labels = [], []

    for image, annotation in zip(images, annotations):
        _, h, w = image.shape

        for box, label in zip(annotation["boxes"], annotation["labels"]):
            box = box.to(torch.int64)
            x1 = max(box[0] - padding, 0)
            y1 = max(box[1] - padding, 0)
            x2 = min(box[2] + padding, w - 1)
            y2 = min(box[3] + padding, h - 1)

            if x1 == x2 or y1 == y2:   # zero-size box, cannot be resized
                continue

            crops.append(TF.resize(image[:, y1:y2, x1:x2], [crop_size, crop_size], antialias=True))
            labels.append(label)

        for _ in range(backgrounds_per_image):
            background = get_background_crop(image, annotation["boxes"].to(torch.int64))
            if background is None:
                break
            crops.append(TF.resize(background, [crop_size, crop_size], antialias=True))
            labels.append(torch.tensor(0))

    return crops, labels
