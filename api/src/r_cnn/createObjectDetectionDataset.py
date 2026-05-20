from pathlib import Path
from tqdm import tqdm
from ..util import iou
from .cocoDetectionDataset import CocoDetectionDataset
from torchvision.transforms import ToPILImage, ToTensor
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import random



SAVE_PATH = "./saved_models"

def get_transform():
    return ToTensor()

dataset = CocoDetectionDataset(
    image_dir="data/faces/train", 
    annotation_path="data/faces/train/_annotations.coco.json",
    transforms=get_transform()
)

processing_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

def random_crop(W, H, sizeX=64, sizeY =64):
    x = random.randint(0, max(W - sizeX, 0))
    y = random.randint(0, max(H - sizeY, 0))
    return x, y, x + sizeX, y + sizeY

def box_inside(inner_box, outer_box):
    return (
        inner_box[0] >= outer_box[0] and
        inner_box[1] >= outer_box[1] and
        inner_box[2] <= outer_box[2] and
        inner_box[3] <= outer_box[3]
    )

def is_background(crop, gt_boxes, threshold=0.0):
    for box in gt_boxes:
        if iou(crop, box) > threshold:
            return False
    return True

def get_background_crop(image, gt_boxes, max_trials=100):
    C, H, W = image.shape

    for _ in range(max_trials):
        x1, y1, x2, y2 = random_crop(W, H, sizeX=random.randint(50, 600), sizeY=random.randint(50, 600))

        crop_box = (x1, y1, x2, y2)

        if is_background(crop_box, gt_boxes, threshold=0.0):
            crop = image[:, y1:y2, x1:x2]
            return crop

    return None

def create_background_tensor(amount, dataset, labels, boxes, image):
    for _ in range(amount):
        background = get_background_crop(image=image, gt_boxes=boxes.to(torch.int64))
        if background is None:
            break
        background = TF.resize(background, [64, 64], antialias=True)
        dataset.append(background)
        labels.append(torch.tensor(0))
    return dataset, labels

def create_stack(images, annotations, PADDING):
    croped_images = []
    labels = []
    for i in range(len(images)):
        image = images[i]
        annotation = annotations[i]
        _, h, w = image.shape
        
        for box, label in tuple(zip(annotation["boxes"], annotation["labels"])):
            box = box.to(torch.int64)
            box_copy = []

            box_copy.append(max(box[0]-PADDING, 0))
            box_copy.append(max(box[1]-PADDING, 0))
            box_copy.append(min(box[2]+PADDING, w-1))
            box_copy.append(min(box[3]+PADDING, h-1))

            if box_copy[1] == box_copy[3] or box_copy[0] == box_copy[2]:
                continue

            croped_image = image[: , box_copy[1]:box_copy[3], box_copy[0]:box_copy[2]]
            croped_image = TF.resize(croped_image, [64, 64], antialias=True)
            croped_images.append(croped_image)
            labels.append(label)

        croped_images, labels = create_background_tensor(amount=10, dataset=croped_images, labels=labels, boxes=annotation["boxes"], image=image)
    return croped_images, labels

def create_croped_dataset():
    to_pil = ToPILImage()

    out_path = Path("data/faces_processed/face")
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = Path("data/faces_processed/background")
    out_path.mkdir(parents=True, exist_ok=True)

    count_image = 0
    count_background = 0
    for images, annotations in tqdm(processing_loader):
        croped_images, labels = create_stack(images, annotations, PADDING=0)

        for i in range(len(croped_images)):
            pil_img = to_pil(croped_images[i])

            if labels[i] == 1:
                pil_img.save(f"data/faces_processed/face/{count_image}.jpg")
                count_image += 1
            else:
                pil_img.save(f"data/faces_processed/background/{count_background}.jpg")
                count_background += 1