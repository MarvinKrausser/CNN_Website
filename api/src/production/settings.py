"""All server settings in one place. Values marked (env) can be set with an
environment variable of the same name, e.g. in compose.yaml."""

import os
from pathlib import Path

from dotenv import load_dotenv

API_ROOT = Path(__file__).resolve().parents[2]    # api/ locally, /app in Docker

BUILD_PATH = Path(os.getenv("BUILD_PATH", API_ROOT / "build_models"))     # (env)
DATABASE_DIR = Path(os.getenv("DATABASE_DIR", API_ROOT / "database"))    # (env)
DATABASE = DATABASE_DIR / "reviews.db"

load_dotenv(DATABASE_DIR / ".env")
API_KEY = os.getenv("API_KEY")  # (env) if missing, protected endpoints reject every request

ORIGINS = [
    "https://marvinkrausser.com",
]

# ---- Models ----
BIRD_MODEL_FILE = "bird_cnn"
BIRD_IMAGE_SIZE = 64
BIRD_C_HIDDEN = 16

FACE_MODEL_FILE = "face_detection_yolo"
FACE_IMAGE_SIZE = 64
FACE_GRID = 6
FACE_C_HIDDEN = 16
FACE_SIZE_ACTIVATION = "exp"   # must match how the deployed weights were trained
FACE_THRESHOLD = 0.9

# ---- Resource limits (the server is weak, keep everything small and bounded) ----
# Number of predictions (bird or face) that may run at the same time. (env)
PARALLEL_INFERENCES = max(1, int(os.getenv("PARALLEL_INFERENCES", "1")))
# Extra requests allowed to wait for a free slot; anything beyond is rejected. (env)
QUEUED_INFERENCES = max(0, int(os.getenv("QUEUED_INFERENCES", "1")))
# PyTorch threads used by EACH running prediction. Total CPU use is roughly
# PARALLEL_INFERENCES * TORCH_THREADS, so keep the product <= your CPU cores. (env)
TORCH_THREADS = max(1, int(os.getenv("TORCH_THREADS", "1")))

MAX_UPLOAD_BYTES = int(max(1, float(os.getenv("MAX_UPLOAD_MEGABYTES", "15"))) * 1024 * 1024)  # (env)
MAX_FRAME_BYTES = 1 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
DECODE_DRAFT_SIZE = (256, 256)  # JPEG decodes at reduced scale, still larger than the 64px model input

MAX_WEBSOCKETS = 3
MAX_WEBSOCKETS_PER_IP = 1
FACE_MAX_FPS = float(os.getenv("FACE_MAX_FPS", "5.5"))  # (env) per websocket connection
FACE_MIN_INTERVAL = 1 / FACE_MAX_FPS

# ---- Rate limits: at most LIMIT requests per WINDOW seconds per IP ----
PREDICT_RATE_LIMIT = max(1, int(os.getenv("PREDICT_RATE_LIMIT", "10")))          # (env)
PREDICT_RATE_WINDOW = max(1.0, float(os.getenv("PREDICT_RATE_WINDOW", "60")))    # (env)
REVIEW_RATE_LIMIT = max(1, int(os.getenv("REVIEW_RATE_LIMIT", "5")))             # (env)
REVIEW_RATE_WINDOW = max(1.0, float(os.getenv("REVIEW_RATE_WINDOW", "60")))      # (env)
