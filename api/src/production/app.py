"""FastAPI app. Start with: uvicorn src.production.app:app (from api/),
or: python -m src.main server.run"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.production import predict, reviews, settings
from src.production.inference import load_models

logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    if not settings.API_KEY:
        logger.warning("API_KEY is not set: GET/DELETE /review are disabled")
    load_models()
    reviews.init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(predict.router)
app.include_router(reviews.router)


@app.get("/health")
def health():
    return {"status": "ok"}
