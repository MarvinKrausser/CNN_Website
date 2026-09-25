"""Loading the models and running one prediction. These functions are
blocking and run on the InferenceGate worker threads."""

import io

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.common.boxes import scale_box
from src.common.checkpoints import load_checkpoint
from src.models.bird_cnn import BirdSpecies, Bird_CNN
from src.models.yolo import Yolo_model, convert_prediction
from src.production import settings

Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}

transform_bird = transforms.Compose([
    transforms.Resize(settings.BIRD_IMAGE_SIZE),
    transforms.CenterCrop(settings.BIRD_IMAGE_SIZE),
    transforms.ToTensor()
])

transform_face = transforms.Compose([
    transforms.Resize((settings.FACE_IMAGE_SIZE, settings.FACE_IMAGE_SIZE)),
    transforms.ToTensor()
])


def load_models():
    bird = Bird_CNN(c_in=3, c_hidden=settings.BIRD_C_HIDDEN, c_out=len(BirdSpecies))
    models["bird"] = load_checkpoint(bird, settings.BUILD_PATH / settings.BIRD_MODEL_FILE, device)

    face = Yolo_model(c_in=3, boxes=1, grid=settings.FACE_GRID, labels=1, c_hidden=settings.FACE_C_HIDDEN,
                      size_activation=settings.FACE_SIZE_ACTIVATION)
    models["face"] = load_checkpoint(face, settings.BUILD_PATH / settings.FACE_MODEL_FILE, device)


class InvalidImage(Exception):
    pass


def open_image(data: bytes) -> tuple[Image.Image, tuple[int, int]]:
    """Open lazily, reject oversized images before decoding, decode JPEGs at
    reduced scale. Returns the image and its ORIGINAL (width, height)."""
    try:
        image = Image.open(io.BytesIO(data))
        original_size = image.size
        if original_size[0] * original_size[1] > settings.MAX_IMAGE_PIXELS:
            raise InvalidImage("Image too large")
        image.draft("RGB", settings.DECODE_DRAFT_SIZE)
        return image.convert("RGB"), original_size
    except InvalidImage:
        raise
    except Exception as e:  # PIL raises many different types for bad data
        raise InvalidImage("Not a valid image") from e


def classify_bird(data: bytes) -> dict:
    image, _ = open_image(data)
    tensor = transform_bird(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = F.softmax(models["bird"](tensor), dim=1)
        confidence, cls = torch.max(probs, dim=1)
    return {
        "class": BirdSpecies(cls.item()).name,
        "confidence": confidence.item()
    }


def detect_faces(data: bytes) -> list:
    image, (W, H) = open_image(data)
    scale_w = W / settings.FACE_IMAGE_SIZE
    scale_h = H / settings.FACE_IMAGE_SIZE

    tensor = transform_face(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = models["face"](tensor)
        bboxes, _, _ = convert_prediction(pred.squeeze(0), tensor.squeeze(0), threshold=settings.FACE_THRESHOLD)

    return [scale_box(b, scale_w, scale_h) for b in bboxes]
