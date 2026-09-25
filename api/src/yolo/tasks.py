"""YOLO face detector tasks. Run with: python -m src.main yolo.<task>"""

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes

from src.common.checkpoints import checkpoint_path, load_checkpoint
from src.common.device import get_device
from src.common.training import benchmark_num_workers, fit
from src.common.visualize import show_image
from src.models.yolo import Yolo_model, convert_prediction
from src.yolo.config import Config
from src.yolo.dataset import YoloDataset
from src.yolo.loss import YoloLoss


def build_dataset(cfg: Config, folder: str, augment: bool):
    return YoloDataset(
        image_dir=folder,
        annotation_path=os.path.join(folder, cfg.annotation_file),
        img_size=cfg.image_size,
        transform=augment,
        grid=cfg.grid,
    )


def build_model(cfg: Config):
    return Yolo_model(c_in=3, boxes=cfg.boxes, grid=cfg.grid, labels=cfg.labels,
                      c_hidden=cfg.c_hidden, size_activation=cfg.size_activation)


def load_trained(cfg: Config, device):
    path = cfg.checkpoint or checkpoint_path(cfg.save_dir, cfg.model_name)
    return load_checkpoint(build_model(cfg), path, device)


def draw_prediction(label, image, threshold):
    """Red: detected boxes, blue: cells with an object, green: empty cells."""
    boxes, cells_obj, cells_noobj = convert_prediction(label, image, threshold)
    if len(boxes) == 0:
        print("no boxes")
    image = draw_bounding_boxes(image, torch.tensor(cells_noobj), colors=(0, 255, 0))
    image = draw_bounding_boxes(image, torch.tensor(cells_obj), colors=(0, 0, 255))
    image = draw_bounding_boxes(image, torch.tensor(boxes), colors=(255, 0, 0))
    show_image(image)


def train(cfg: Config):
    """Train the detector; saves the model with the lowest validation loss."""
    device = get_device(cfg.device)
    train_loader = DataLoader(build_dataset(cfg, cfg.train_dir, cfg.augment_train),
                              batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.train_workers)
    val_loader = DataLoader(build_dataset(cfg, cfg.val_dir, cfg.augment_val),
                            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.val_workers)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    fit(model, YoloLoss(), optimizer, train_loader, val_loader, device,
        epochs=cfg.epochs, save_dir=cfg.save_dir, model_name=cfg.model_name, save=cfg.save,
        monitor="loss")


def view_data(cfg: Config):
    """Show validation images with their ground-truth boxes, one at a time."""
    dataset = build_dataset(cfg, cfg.val_dir, cfg.augment_val)
    for images, labels in DataLoader(dataset, batch_size=1, shuffle=True):
        draw_prediction(labels[0], images[0], cfg.threshold)


def sample(cfg: Config):
    """Show the saved model's predictions on validation images."""
    device = get_device(cfg.device)
    model = load_trained(cfg, device)
    dataset = build_dataset(cfg, cfg.val_dir, cfg.augment_val)

    for images, _ in DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False):
        with torch.inference_mode():
            start = time.perf_counter()
            predictions = model(images.to(device)).cpu()
            print(f"Inference: {time.perf_counter() - start:.6f} seconds")
        for image, prediction in zip(images, predictions):
            draw_prediction(prediction, image, cfg.threshold)


def benchmark_workers(cfg: Config):
    """Time DataLoader num_workers values to pick the fastest."""
    device = get_device(cfg.device)
    dataset = build_dataset(cfg, cfg.train_dir, cfg.augment_train)
    benchmark_num_workers(dataset, build_model(cfg).to(device), device, batch_size=cfg.batch_size)


def webcam(cfg: Config):
    """Live detection on the webcam (--camera). Press q to quit."""
    import cv2

    device = get_device(cfg.device)
    model = load_trained(cfg, device)

    cap = cv2.VideoCapture(cfg.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    def draw(frame, boxes, scale_w, scale_h, color, thickness):
        for b in boxes:
            cv2.rectangle(frame, (int(b[0] * scale_w), int(b[1] * scale_h)),
                          (int(b[2] * scale_w), int(b[3] * scale_h)), color, thickness)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            _, H, W = image.shape
            scale_w, scale_h = W / cfg.image_size, H / cfg.image_size

            image = F.interpolate(image.unsqueeze(0), size=(cfg.image_size, cfg.image_size),
                                  mode="bilinear", align_corners=True)
            with torch.inference_mode():
                prediction = model(image.to(device)).cpu()

            boxes, cells_obj, cells_noobj = convert_prediction(prediction.squeeze(0), image.squeeze(0),
                                                               threshold=cfg.threshold)
            draw(frame, cells_noobj, scale_w, scale_h, (0, 255, 0), 1)
            draw(frame, cells_obj, scale_w, scale_h, (255, 0, 0), 1)
            draw(frame, boxes, scale_w, scale_h, (0, 0, 255), 4)

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


TASKS = {
    "train": train,
    "view-data": view_data,
    "sample": sample,
    "benchmark-workers": benchmark_workers,
    "webcam": webcam,
}
