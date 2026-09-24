import asyncio
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from enum import Enum
import io
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from torchvision import transforms

from yolo_model_production import convert_prediction, Yolo_model
from bird_cnn_production import Bird_CNN

logger = logging.getLogger("server")

BASE_DIR = Path(__file__).resolve().parent
BUILD_PATH = Path(os.getenv("BUILD_PATH", BASE_DIR / "build_models"))
DATABASE_DIR = Path(os.getenv("DATABASE_DIR", BASE_DIR / "database"))
DATABASE = DATABASE_DIR / "reviews.db"

load_dotenv(DATABASE_DIR / ".env")
API_KEY = os.getenv("API_KEY")  # if missing, protected endpoints reject every request

IMAGE_SIZE_CNN = 64
IMAGE_SIZE_YOLO = 64

# ---- Resource limits (the server is weak, keep everything small and bounded) ----
# Number of predictions (bird or face) that may run at the same time.
PARALLEL_INFERENCES = max(1, int(os.getenv("PARALLEL_INFERENCES", "1")))
# Extra requests allowed to wait for a free slot; anything beyond is rejected.
QUEUED_INFERENCES = max(0, int(os.getenv("QUEUED_INFERENCES", "1")))
# PyTorch threads used by EACH running prediction. Total CPU use is roughly
# PARALLEL_INFERENCES * TORCH_THREADS, so keep the product <= your CPU cores.
TORCH_THREADS = max(1, int(os.getenv("TORCH_THREADS", "1")))
MAX_UPLOAD_BYTES = int(max(1, float(os.getenv("MAX_UPLOAD_MEGABYTES", "15"))) * 1024 * 1024)
MAX_FRAME_BYTES = 1 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_WEBSOCKETS = 3
MAX_WEBSOCKETS_PER_IP = 1
FACE_MAX_FPS = float(os.getenv("FACE_MAX_FPS", "5.5"))  # per websocket connection
FACE_MIN_INTERVAL = 1 / FACE_MAX_FPS
DECODE_DRAFT_SIZE = (256, 256)  # JPEG decodes at reduced scale, still larger than the 64px model input

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

origins = [
    "https://marvinkrausser.com",
]


class BirdSpecies(Enum):
    Common_Kingfisher = 0
    Common_Myna = 1
    House_Crow = 2
    Indian_Peacock = 3
    Indian_Pitta = 4
    Ruddy_Shelduck = 5
    Sarus_Crane = 6


transform_bird = transforms.Compose([
    transforms.Resize(IMAGE_SIZE_CNN),
    transforms.CenterCrop(IMAGE_SIZE_CNN),
    transforms.ToTensor()
])

transform_face = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_YOLO, IMAGE_SIZE_YOLO)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}


def load_models():
    model_bird = Bird_CNN(c_in=3, c_hidden=16, c_out=7)
    model_bird.load_state_dict(torch.load(BUILD_PATH / "bird_cnn", map_location=device, weights_only=True))
    model_bird.to(device).eval()

    model_face = Yolo_model(c_in=3, boxes=1, grid=6, labels=1, c_hidden=16)
    model_face.load_state_dict(torch.load(BUILD_PATH / "face_detection_yolo", map_location=device, weights_only=True))
    model_face.to(device).eval()

    models["bird"] = model_bird
    models["face"] = model_face


def init_db():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                website TEXT NOT NULL,
                rating INTEGER NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    if not API_KEY:
        logger.warning("API_KEY is not set: GET/DELETE /review are disabled")
    load_models()
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# --------------------------------------------------------------------------
# Overload protection
# --------------------------------------------------------------------------

def client_ip(conn: Request | WebSocket) -> str:
    # Caddy is the only proxy in front of this container and appends the real
    # client address as the last X-Forwarded-For entry.
    forwarded = conn.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return conn.client.host if conn.client else "unknown"


class RateLimiter:
    """Sliding-window limiter per key (in memory, single process)."""

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self.hits = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if len(self.hits) > 10_000:
            self.hits = defaultdict(deque, {
                k: v for k, v in self.hits.items() if v and now - v[-1] < self.window
            })
        q = self.hits[key]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


predict_limiter = RateLimiter(
    limit=max(1, int(os.getenv("PREDICT_RATE_LIMIT", "10"))),
    window=max(1.0, float(os.getenv("PREDICT_RATE_WINDOW", "60"))),
)
review_limiter = RateLimiter(
    limit=max(1, int(os.getenv("REVIEW_RATE_LIMIT", "5"))),
    window=max(1.0, float(os.getenv("REVIEW_RATE_WINDOW", "60"))),
)


def rate_limit(limiter: RateLimiter):
    def dependency(request: Request):
        if not limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="Too many requests", headers={"Retry-After": "60"})
    return dependency


class Busy(Exception):
    pass


def init_inference_thread():
    # Must run inside each worker thread: with OpenMP the thread count is a
    # per-thread setting, so setting it once in the main thread is not enough.
    torch.set_num_threads(TORCH_THREADS)


class InferenceGate:
    """Runs predictions on `parallel` worker threads. At most
    `parallel + queued` jobs (running + waiting) are admitted; everything
    else is rejected at once instead of queueing up and eating memory.

    The models are in eval mode under torch.inference_mode(), so several
    threads can safely run forward passes on the same model object."""

    def __init__(self, parallel: int, queued: int):
        self.max_pending = parallel + queued
        self.pending = 0
        self.executor = ThreadPoolExecutor(
            max_workers=parallel,
            thread_name_prefix="inference",
            initializer=init_inference_thread,
        )

    async def run(self, fn, *args):
        if self.pending >= self.max_pending:
            raise Busy
        self.pending += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)
        finally:
            self.pending -= 1


gate = InferenceGate(PARALLEL_INFERENCES, QUEUED_INFERENCES)


class InvalidImage(Exception):
    pass


def open_image(data: bytes) -> tuple[Image.Image, tuple[int, int]]:
    """Open lazily, reject oversized images before decoding, decode JPEGs at
    reduced scale. Returns the image and its ORIGINAL (width, height)."""
    try:
        image = Image.open(io.BytesIO(data))
        original_size = image.size
        if original_size[0] * original_size[1] > MAX_IMAGE_PIXELS:
            raise InvalidImage("Image too large")
        image.draft("RGB", DECODE_DRAFT_SIZE)
        return image.convert("RGB"), original_size
    except InvalidImage:
        raise
    except Exception as e:  # PIL raises many different types for bad data
        raise InvalidImage("Not a valid image") from e


# --------------------------------------------------------------------------
# Bird classification
# --------------------------------------------------------------------------

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


async def read_limited(file: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(64 * 1024):
        data += chunk
        if len(data) > limit:
            raise HTTPException(status_code=413, detail="File too large")
    return bytes(data)


@app.post("/predict", dependencies=[Depends(rate_limit(predict_limiter))])
async def predict(file: UploadFile = File(...)):
    data = await read_limited(file, MAX_UPLOAD_BYTES)
    try:
        return await gate.run(classify_bird, data)
    except Busy:
        raise HTTPException(status_code=503, detail="Server busy, try again shortly", headers={"Retry-After": "5"})
    except InvalidImage as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------------------------
# Face detection (websocket)
# --------------------------------------------------------------------------

def detect_faces(data: bytes) -> list:
    image, (W, H) = open_image(data)
    scale_w = W / IMAGE_SIZE_YOLO
    scale_h = H / IMAGE_SIZE_YOLO

    tensor = transform_face(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred = models["face"](tensor)
        bboxes, _, _ = convert_prediction(pred.squeeze(0), tensor.squeeze(0), threshold=0.9)

    return [
        [int(b[0] * scale_w), int(b[1] * scale_h), int(b[2] * scale_w), int(b[3] * scale_h)]
        for b in bboxes
    ]


active_websockets = defaultdict(int)


@app.websocket("/predict_face")
async def predict_face(websocket: WebSocket):
    ip = client_ip(websocket)

    # CORS does not apply to websockets, so check the origin ourselves.
    if websocket.headers.get("origin") not in origins:
        await websocket.close(code=1008)
        return
    if (sum(active_websockets.values()) >= MAX_WEBSOCKETS
            or active_websockets[ip] >= MAX_WEBSOCKETS_PER_IP):
        await websocket.close(code=1013)  # try again later
        return

    active_websockets[ip] += 1
    try:
        await websocket.accept()
        last = 0.0

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None or len(data) > MAX_FRAME_BYTES:
                await websocket.close(code=1009 if data else 1003)
                break

            # Drop frames that arrive too fast or while the server is busy
            # instead of queueing them.
            now = time.monotonic()
            if now - last < FACE_MIN_INTERVAL:
                continue
            last = now

            try:
                boxes = await gate.run(detect_faces, data)
            except Busy:
                continue
            except InvalidImage:
                await websocket.close(code=1003)
                break

            await websocket.send_json({"bboxes": boxes})

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("predict_face failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        active_websockets[ip] -= 1
        if active_websockets[ip] <= 0:
            del active_websockets[ip]


# --------------------------------------------------------------------------
# Reviews
# --------------------------------------------------------------------------

def get_api_key(authorization: str = Header(None)):
    if not API_KEY or not authorization or not secrets.compare_digest(
        authorization.encode(), f"Bearer {API_KEY}".encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return authorization


class Review(BaseModel):
    website: str = Field(min_length=1, max_length=100)
    rating: int = Field(ge=1, le=5)
    text: str = Field(max_length=1000)
    # The client also sends a "date"; it is ignored, the server sets the time.


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/review")
def get_reviews(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth=Depends(get_api_key),
):
    with closing(connect_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM reviews ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    return {"data": [dict(row) for row in rows]}


# Public on purpose: no credentials needed, only rate limited per IP.
@app.post("/review", dependencies=[Depends(rate_limit(review_limiter))])
def post_review(review: Review):
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(connect_db()) as conn:
        conn.execute(
            "INSERT INTO reviews (website, rating, text, created_at) VALUES (?, ?, ?, ?)",
            (review.website, review.rating, review.text.strip(), created_at),
        )
        conn.commit()
    return {"status": "ok"}


@app.delete("/review")
def delete_review(auth=Depends(get_api_key)):
    with closing(connect_db()) as conn:
        conn.execute("DELETE FROM reviews")
        conn.commit()
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}
