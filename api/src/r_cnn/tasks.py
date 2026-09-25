"""R-CNN style experiments. Run with: python -m src.main rcnn.<task>"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

from src.common.checkpoints import checkpoint_path, load_checkpoint
from src.common.data import collate_as_tuples, split_dataset
from src.common.device import get_device
from src.common.training import fit
from src.models.r_cnn import ObjectDetectionCNN
from src.r_cnn.config import Config
from src.r_cnn.crops import create_stack
from src.r_cnn.dataset import CocoDetectionDataset
from src.r_cnn.detect import detect


def build_model(cfg: Config):
    return ObjectDetectionCNN(c_in=3, c_hidden=cfg.c_hidden, c_out=cfg.num_classes, layers=cfg.layers)


def coco_dataset(cfg: Config, folder: str):
    return CocoDetectionDataset(image_dir=folder,
                                annotation_path=os.path.join(folder, cfg.annotation_file),
                                transforms=ToTensor())


def create_dataset(cfg: Config):
    """Cut face and background crops out of --faces-raw-dir (for train-faces)."""
    loader = DataLoader(coco_dataset(cfg, cfg.faces_raw_dir), batch_size=1, shuffle=False,
                        collate_fn=collate_as_tuples)
    to_pil = ToPILImage()
    out = Path(cfg.faces_processed_dir)
    (out / "face").mkdir(parents=True, exist_ok=True)
    (out / "background").mkdir(parents=True, exist_ok=True)

    counts = {"face": 0, "background": 0}
    for images, annotations in tqdm(loader):
        crops, labels = create_stack(images, annotations, padding=0, crop_size=cfg.crop_size,
                                     backgrounds_per_image=cfg.backgrounds_per_image)
        for crop, label in zip(crops, labels):
            kind = "face" if label == 1 else "background"
            to_pil(crop).save(out / kind / f"{counts[kind]}.jpg")
            counts[kind] += 1

    print(f"Saved {counts['face']} faces and {counts['background']} backgrounds to {out}")


def train_faces(cfg: Config):
    """Train the face/background crop classifier on --faces-processed-dir."""
    device = get_device(cfg.device)

    base = [transforms.Resize(cfg.crop_size), transforms.CenterCrop(cfg.crop_size), transforms.ToTensor()]
    augment = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=15)] + base
    train_ds, val_ds = split_dataset(datasets.ImageFolder(cfg.faces_processed_dir), cfg.train_fraction,
                                     transforms.Compose(augment), transforms.Compose(base))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # ImageFolder sorts classes alphabetically: 0 = background, 1 = face
    weights = torch.tensor([cfg.background_weight, 1.0], device=device)

    fit(model, nn.CrossEntropyLoss(weight=weights), optimizer, train_loader, val_loader, device,
        epochs=cfg.epochs, save_dir=cfg.save_dir, model_name=cfg.faces_model_name, save=cfg.save,
        monitor="loss", track_accuracy=True)


def detect_faces(cfg: Config):
    """Run selective search + the face classifier on --image."""
    if not cfg.image:
        raise SystemExit("Pass the image to run on with --image path/to/image.jpg")
    device = get_device(cfg.device)
    model = load_checkpoint(build_model(cfg), checkpoint_path(cfg.save_dir, cfg.faces_model_name), device)
    image = read_image(cfg.image).float() / 255.0
    detect(model, image, device, crop_size=cfg.crop_size, min_size=cfg.min_size,
           max_size=cfg.max_size, min_confidence=cfg.min_confidence)


def train_football(cfg: Config):
    """Train the crop classifier on the football COCO dataset (crops on the fly)."""
    device = get_device(cfg.device)
    train_loader = DataLoader(coco_dataset(cfg, cfg.football_train_dir), batch_size=cfg.football_batch_size,
                              shuffle=True, collate_fn=collate_as_tuples)
    val_loader = DataLoader(coco_dataset(cfg, cfg.football_val_dir), batch_size=cfg.football_batch_size,
                            shuffle=True, collate_fn=collate_as_tuples)

    def prepare_batch(batch, device):
        images, annotations = batch
        crops, labels = create_stack(images, annotations, padding=cfg.padding, crop_size=cfg.crop_size)
        return torch.stack(crops).to(device), torch.stack(labels).to(device)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    fit(model, nn.CrossEntropyLoss(), optimizer, train_loader, val_loader, device,
        epochs=cfg.epochs, save_dir=cfg.save_dir, model_name=cfg.football_model_name, save=cfg.save,
        monitor="loss", track_accuracy=True, prepare_batch=prepare_batch)


def detect_football(cfg: Config):
    """Run selective search + the football classifier on a validation image."""
    device = get_device(cfg.device)
    model = load_checkpoint(build_model(cfg), checkpoint_path(cfg.save_dir, cfg.football_model_name), device)
    loader = DataLoader(coco_dataset(cfg, cfg.football_val_dir), batch_size=1, shuffle=True,
                        collate_fn=collate_as_tuples)
    image = next(iter(loader))[0][0]
    detect(model, image, device, padding=cfg.padding, crop_size=cfg.crop_size, min_size=cfg.min_size,
           max_size=cfg.football_max_size, min_confidence=cfg.football_min_confidence)


TASKS = {
    "create-dataset": create_dataset,
    "train-faces": train_faces,
    "detect-faces": detect_faces,
    "train-football": train_football,
    "detect-football": detect_football,
}
