"""Records every training run in its own folder, so runs with different
settings or architectures can be compared later.

    <runs_dir>/<model_name>/<YYYYmmdd-HHMMSS>[_<run_name>]/
        run.json          everything about the run: config, command, model
                          (class, parameter counts, architecture hash),
                          optimizer, loss, data sizes, environment, git state
                          and the result (status, best epoch and metric,
                          duration). Rewritten after every epoch.
        metrics.csv       one row per epoch
        architecture.txt  the model's module tree, str(model)
        changes.diff      uncommitted code changes (only if there are any)
        best.pt           state_dict of the best epoch so far

result.status in run.json is "running" during training, then "completed",
"failed" or "interrupted". A run whose process was killed stays "running".
"""

import csv
import dataclasses
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True,
                              timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.tolist() if value.numel() <= 16 else f"tensor{tuple(value.shape)}"
    return str(value)


def _module_settings(module):
    """Public attributes and small buffers, e.g. a loss's reduction and class weights."""
    settings = {k: v for k, v in vars(module).items() if not k.startswith("_") and k != "training"}
    settings.update(dict(module.named_buffers()))
    return _jsonable(settings)


def _dataset_size(loader):
    try:
        return len(loader.dataset)
    except TypeError:
        return None


class RunLog:
    def __init__(self, runs_dir, model_name, config, model, optimizer, loss_fn,
                 train_loader, val_loader, device, monitor):
        self.monitor = monitor
        self.started = time.time()
        self.dir = self._make_dir(Path(runs_dir) / model_name, getattr(config, "run_name", ""))
        self._csv_fields = None

        architecture = str(model)
        status = _git("status", "--porcelain")
        diff = _git("diff", "HEAD")
        gpu = torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None

        self.info = {
            "id": f"{model_name}/{self.dir.name}",
            "model_name": model_name,
            "started": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "config": _jsonable(dataclasses.asdict(config)),
            "model": {
                "class": type(model).__name__,
                "params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "architecture_hash": hashlib.sha256(architecture.encode()).hexdigest()[:12],
            },
            "optimizer": {
                "class": type(optimizer).__name__,
                **_jsonable({k: v for k, v in optimizer.param_groups[0].items() if k != "params"}),
            },
            "loss": {"class": type(loss_fn).__name__, **_module_settings(loss_fn)},
            "data": {
                "train_size": _dataset_size(train_loader),
                "val_size": _dataset_size(val_loader),
                "batch_size": train_loader.batch_size,
                "train_workers": train_loader.num_workers,
            },
            "environment": {
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "gpu": gpu,
                "seed": torch.initial_seed(),
            },
            "git": {
                "commit": (_git("rev-parse", "HEAD") or "").strip() or None,
                "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
                "dirty": bool(status.strip()) if status is not None else None,
                "changed_files": [line[3:] for line in status.splitlines()] if status else [],
            },
            "result": {
                "status": "running",
                "monitor": monitor,
                "epochs_done": 0,
                "best_epoch": None,
                f"best_val_{monitor}": None,
                "last": None,
                "duration_s": 0,
                "finished": None,
                "error": None,
            },
        }

        (self.dir / "architecture.txt").write_text(architecture + "\n")
        if diff:
            (self.dir / "changes.diff").write_text(diff)
        self._write_info()
        print(f"Logging run to {self.dir}")

    @staticmethod
    def _make_dir(parent, run_name):
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        if run_name:
            name += "_" + re.sub(r"[^\w.-]+", "-", run_name)
        parent.mkdir(parents=True, exist_ok=True)
        for i in range(1, 100):
            path = parent / (name if i == 1 else f"{name}-{i}")
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue
        raise FileExistsError(f"no free run folder for {parent / name}")

    def _write_info(self):
        # Write-then-rename so a reader never sees a half written file.
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(self.info, indent=2) + "\n")
        os.replace(tmp, self.dir / "run.json")

    def log_epoch(self, epoch, train, val, lr, seconds, improved):
        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in train.items()},
               **{f"val_{k}": v for k, v in val.items()},
               "lr": lr, "seconds": round(seconds, 2), "improved": improved}
        path = self.dir / "metrics.csv"
        with path.open("a", newline="") as f:
            if self._csv_fields is None:
                self._csv_fields = list(row)
                writer = csv.DictWriter(f, fieldnames=self._csv_fields)
                writer.writeheader()
            else:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields)
            writer.writerow(row)

        result = self.info["result"]
        result["epochs_done"] = epoch
        result["last"] = {k: v for k, v in row.items() if k != "improved"}
        if improved:
            result["best_epoch"] = epoch
            result[f"best_val_{self.monitor}"] = val[self.monitor]
        result["duration_s"] = round(time.time() - self.started, 1)
        self._write_info()

    def save_weights(self, model):
        torch.save(model.state_dict(), self.dir / "best.pt")

    def finish(self, status, error=None):
        result = self.info["result"]
        result["status"] = status
        result["error"] = error
        result["duration_s"] = round(time.time() - self.started, 1)
        result["finished"] = datetime.now().isoformat(timespec="seconds")
        self._write_info()
