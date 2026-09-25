"""API server tasks. Run with: python -m src.main server.<task>"""

from src.production.config import Config


def run(cfg: Config):
    """Start the API server with uvicorn."""
    import uvicorn

    uvicorn.run("src.production.app:app", host=cfg.host, port=cfg.port,
                reload=cfg.reload, ws_max_size=cfg.ws_max_size)


TASKS = {
    "run": run,
}
