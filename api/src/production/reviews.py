"""Routes and storage for reviews (SQLite)."""

from contextlib import closing
from datetime import datetime, timezone
import secrets
import sqlite3

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from src.production import settings
from src.production.limits import rate_limit, review_limiter

router = APIRouter()


def init_db():
    settings.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(settings.DATABASE)) as conn:
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


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DATABASE, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def get_api_key(authorization: str = Header(None)):
    if not settings.API_KEY or not authorization or not secrets.compare_digest(
        authorization.encode(), f"Bearer {settings.API_KEY}".encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return authorization


class Review(BaseModel):
    website: str = Field(min_length=1, max_length=100)
    rating: int = Field(ge=1, le=5)
    text: str = Field(max_length=1000)
    # The client also sends a "date"; it is ignored, the server sets the time.


@router.get("/review")
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
@router.post("/review", dependencies=[Depends(rate_limit(review_limiter))])
def post_review(review: Review):
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(connect_db()) as conn:
        conn.execute(
            "INSERT INTO reviews (website, rating, text, created_at) VALUES (?, ?, ?, ?)",
            (review.website, review.rating, review.text.strip(), created_at),
        )
        conn.commit()
    return {"status": "ok"}


@router.delete("/review")
def delete_review(auth=Depends(get_api_key)):
    with closing(connect_db()) as conn:
        conn.execute("DELETE FROM reviews")
        conn.commit()
    return {"status": "ok"}
