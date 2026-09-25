from pathlib import Path

import torch


def checkpoint_path(save_dir: str | Path, model_name: str) -> Path:
    # Layout kept from the original code: <save_dir>/<name>/<name>
    return Path(save_dir) / model_name / model_name


def save_checkpoint(model: torch.nn.Module, save_dir: str | Path, model_name: str) -> Path:
    path = checkpoint_path(save_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
