"""Selective search + crop classifier on a full image."""

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm


def resize_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / new_w, h / new_h


def detect(model, image, device, padding=0, crop_size=64, min_size=5, max_size=600, min_confidence=0.8):
    """image: tensor [C, H, W] in 0..1. Shows the image, then the detections.
    Needs opencv-contrib-python (cv2.ximgproc)."""
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("Selective search needs opencv-contrib-python (cv2.ximgproc)")

    _, H, W = image.shape
    image_numpy = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    resized, scale_w, scale_h = resize_keep_aspect(image_numpy, W, H)

    cv2.imshow("", resized)
    cv2.waitKey(0)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(resized)
    ss.switchToSelectiveSearchFast()

    predictions = []
    with torch.inference_mode():
        for (x, y, w, h) in tqdm(ss.process()):
            x, y, w, h = int(x * scale_w), int(y * scale_h), int(w * scale_w), int(h * scale_h)
            if w < min_size or h < min_size or w > max_size or h > max_size:
                continue

            x1, y1 = max(x - padding, 0), max(y - padding, 0)
            x2, y2 = min(x + w + padding, W - 1), min(y + h + padding, H - 1)

            crop = TF.resize(image[:, y1:y2, x1:x2], [crop_size, crop_size], antialias=True)
            probs = F.softmax(model(crop.unsqueeze(0).to(device)), dim=1)
            confidence, cls = torch.max(probs, dim=1)
            if cls.item() == 1 and confidence.item() > min_confidence:
                predictions.append((confidence.item(), (x, y, w, h)))

    print(f"{len(predictions)} detections")
    draw = image_numpy.copy()
    for _, (x, y, w, h) in sorted(predictions, reverse=True):
        cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow("", draw)
    cv2.waitKey(0)
