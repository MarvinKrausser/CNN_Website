from dataclasses import dataclass


@dataclass
class Config:
    """How to start the API server locally. Server limits (rate limits,
    parallel inferences, ...) are in src/production/settings.py and are set
    with environment variables."""

    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    ws_max_size: int = 1024 * 1024   # largest websocket message in bytes
