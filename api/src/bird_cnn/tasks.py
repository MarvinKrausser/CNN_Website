"""Bird classifier tasks. Run with: python -m src.main bird.<task>"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.bird_cnn.config import Config
from src.common.checkpoints import checkpoint_path, load_checkpoint
from src.common.data import split_dataset
from src.common.device import get_device
from src.common.training import fit
from src.common.visualize import show_batch
from src.models.bird_cnn import BirdSpecies, Bird_CNN


def eval_transform(cfg: Config):
    return transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor()
    ])


def train_transform(cfg: Config):
    return transforms.Compose([
        transforms.RandomAffine(degrees=35, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor()
    ])


def build_loaders(cfg: Config):
    dataset = datasets.ImageFolder(cfg.data_dir)
    train_ds, val_ds = split_dataset(dataset, cfg.train_fraction, train_transform(cfg), eval_transform(cfg))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader


def build_model(cfg: Config):
    return Bird_CNN(c_in=3, c_hidden=cfg.c_hidden, c_out=cfg.num_classes)


def train(cfg: Config):
    """Train the classifier; saves the model with the best validation accuracy."""
    device = get_device(cfg.device)
    train_loader, val_loader = build_loaders(cfg)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    fit(model, nn.CrossEntropyLoss(), optimizer, train_loader, val_loader, device,
        epochs=cfg.epochs, save_dir=cfg.save_dir, model_name=cfg.model_name, save=cfg.save,
        monitor="acc", track_accuracy=True)


def view_data(cfg: Config):
    """Show a few augmented training images with their labels."""
    train_loader, _ = build_loaders(cfg)
    show_batch(train_loader)


def predict(cfg: Config):
    """Classify one image (--image) with a saved model."""
    device = get_device(cfg.device)
    path = cfg.checkpoint or checkpoint_path(cfg.save_dir, cfg.model_name)
    model = load_checkpoint(build_model(cfg), path, device)

    image = eval_transform(cfg)(Image.open(cfg.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        confidence, pred = torch.max(F.softmax(model(image), dim=1), dim=1)

    cls = pred.item()
    name = BirdSpecies(cls).name if cfg.num_classes == len(BirdSpecies) else f"class {cls}"
    print(f"Species: {name} | Confidence: {confidence.item():.2f}")


TASKS = {
    "train": train,
    "view-data": view_data,
    "predict": predict,
}
