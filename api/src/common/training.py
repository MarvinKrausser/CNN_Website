"""One training loop for all models.

Each epoch trains, validates, prints the metrics and saves the model when the
monitored metric improves. `prepare_batch` turns a loader batch into
(inputs, targets); the default just moves both to the device."""

import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.checkpoints import save_checkpoint


def _default_prepare(batch, device):
    inputs, targets = batch
    return inputs.to(device), targets.to(device)


def _run_epoch(model, loss_fn, loader, device, prepare_batch, track_accuracy, optimizer=None, desc=""):
    training = optimizer is not None
    model.train(training)

    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, targets = prepare_batch(batch, device)

            preds = model(inputs)
            loss = loss_fn(preds, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()
            count += inputs.size(0)
            if track_accuracy:
                correct += (preds.argmax(dim=1) == targets).sum().item()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {"loss": loss_sum / max(count, 1)}
    if track_accuracy:
        metrics["acc"] = correct / max(count, 1)
    return metrics


def fit(
    model,
    loss_fn,
    optimizer,
    train_loader,
    val_loader,
    device,
    epochs: int,
    save_dir: str,
    model_name: str,
    save: bool = True,
    monitor: str = "loss",          # "loss" (lower is better) or "acc" (higher is better)
    track_accuracy: bool = False,   # for classifiers
    prepare_batch=_default_prepare,
):
    best = math.inf if monitor == "loss" else -math.inf

    for epoch in range(1, epochs + 1):
        train = _run_epoch(model, loss_fn, train_loader, device, prepare_batch, track_accuracy,
                           optimizer=optimizer, desc=f"Train {epoch}")
        val = _run_epoch(model, loss_fn, val_loader, device, prepare_batch, track_accuracy,
                         desc=f"Validate {epoch}")

        improved = val[monitor] < best if monitor == "loss" else val[monitor] > best
        if improved:
            best = val[monitor]
            if save:
                save_checkpoint(model, save_dir, model_name)

        # Loss is the summed batch loss per sample, shown x1000.
        parts = [f"epoch: {epoch}"]
        if track_accuracy:
            parts.append(f"train acc: {train['acc'] * 100:.1f}% | val acc: {val['acc'] * 100:.1f}%")
        parts.append(f"train loss: {train['loss'] * 1000:.2f} | val loss: {val['loss'] * 1000:.2f}")
        if improved and save:
            parts.append("saved")
        print(" | ".join(parts))

    return best


def benchmark_num_workers(dataset, model, device, batch_size=64, rounds=2):
    """Prints how long `rounds` passes over the dataset take for different
    DataLoader num_workers values, to pick the fastest one."""
    import multiprocessing as mp

    model.eval()
    for num_workers in range(0, mp.cpu_count(), 2):
        loader = DataLoader(dataset, shuffle=True, num_workers=num_workers,
                            batch_size=batch_size, pin_memory=True)
        start = time.time()
        with torch.no_grad():
            for _ in range(rounds):
                for images, _ in tqdm(loader, leave=False):
                    model(images.to(device))
        print(f"num_workers={num_workers}: {time.time() - start:.0f} seconds")
