"""One training loop for all models.

Each epoch trains, validates, prints the metrics and saves the model when the
monitored metric improves. `prepare_batch` turns a loader batch into
(inputs, targets); the default just moves both to the device.

An optional `scheduler` is stepped once after every epoch, so the learning
rate logged for an epoch is the one it trained with.

Passing the task's `config` also records the run (settings, architecture,
metrics per epoch, best weights) under config.runs_dir, see run_log.py."""

import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.checkpoints import save_checkpoint
from src.common.run_log import RunLog


def _default_prepare(batch, device):
    inputs, targets = batch
    return inputs.to(device), targets.to(device)


def adamw(model, lr, weight_decay):
    """AdamW that decays only conv/linear weights. Biases and BatchNorm
    parameters (everything 1-D) are left alone, shrinking them only hurts."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def warmup_cosine(optimizer, epochs, warmup_epochs, min_lr):
    """Per-epoch schedule: linear warmup from lr/10 over warmup_epochs, then
    cosine decay from lr to min_lr over the remaining epochs."""
    floor = min_lr / optimizer.param_groups[0]["lr"]
    decay_epochs = max(epochs - warmup_epochs - 1, 1)

    def factor(epoch):  # epoch counts from 0
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        progress = min(epoch - warmup_epochs, decay_epochs) / decay_epochs
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _run_epoch(model, loss_fn, loader, device, prepare_batch, track_accuracy, optimizer=None,
               val_metrics=None, desc=""):
    training = optimizer is not None
    model.train(training)

    loss_sum, correct, count = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, targets = prepare_batch(batch, device)

            preds = model(inputs)

            if training:
                loss = loss_fn(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * inputs.size(0)
            else:
                all_preds.append(preds)
                all_targets.append(targets)

            count += inputs.size(0)
            if track_accuracy:
                correct += (preds.argmax(dim=1) == targets).sum().item()

    if training:
        metrics = {"loss": loss_sum / max(count, 1)}
    else:
        # Validation loss over the whole set at once, so it doesn't depend on
        # the batch size (YoloLoss averages per batch over the cells with a face).
        preds, targets = torch.cat(all_preds), torch.cat(all_targets)
        metrics = {"loss": loss_fn(preds, targets).item()}
        if val_metrics:
            metrics.update(val_metrics(preds, targets))
        del preds, targets, all_preds, all_targets
    if track_accuracy:
        metrics["acc"] = correct / max(count, 1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    monitor: str = "loss",          # "loss" (lower is better) or any other val metric (higher is better)
    track_accuracy: bool = False,   # for classifiers
    prepare_batch=_default_prepare,
    val_metrics=None,               # f(preds, targets) -> dict of extra metrics over the whole val set
    config=None,                    # the task's Config; enables the run log
    scheduler=None,                 # LR scheduler, stepped once per epoch
    save_every_epoch: bool = False, # with save: also keep every epoch as <model_name>_epochNNN
):
    run = None
    if config is not None and config.runs_dir:
        run = RunLog(config.runs_dir, model_name, config, model, optimizer, loss_fn,
                     train_loader, val_loader, device, monitor)
    try:
        best = _fit_epochs(model, loss_fn, optimizer, train_loader, val_loader, device, epochs,
                           save_dir, model_name, save, monitor, track_accuracy, prepare_batch, val_metrics, run,
                           scheduler, save_every_epoch)
    except KeyboardInterrupt:
        if run:
            run.finish("interrupted")
        raise
    except Exception as e:
        if run:
            run.finish("failed", error=repr(e))
        raise
    if run:
        run.finish("completed")
    return best


def _fit_epochs(model, loss_fn, optimizer, train_loader, val_loader, device, epochs,
                save_dir, model_name, save, monitor, track_accuracy, prepare_batch, val_metrics, run,
                scheduler, save_every_epoch):
    best = math.inf if monitor == "loss" else -math.inf

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train = _run_epoch(model, loss_fn, train_loader, device, prepare_batch, track_accuracy,
                           optimizer=optimizer, desc=f"Train {epoch}")
        val = _run_epoch(model, loss_fn, val_loader, device, prepare_batch, track_accuracy,
                         val_metrics=val_metrics, desc=f"Validate {epoch}")

        improved = val[monitor] < best if monitor == "loss" else val[monitor] > best
        if improved:
            best = val[monitor]
            if save:
                save_checkpoint(model, save_dir, model_name)
            if run:
                run.save_weights(model)
        if save and save_every_epoch:
            save_checkpoint(model, save_dir, model_name, epoch=epoch)
        if run:
            run.log_epoch(epoch, train, val, lr=optimizer.param_groups[0]["lr"],
                          seconds=time.perf_counter() - start, improved=improved)
        if scheduler:
            scheduler.step()

        parts = [f"epoch: {epoch}"]
        if track_accuracy:
            parts.append(f"train acc: {train['acc'] * 100:.1f}% | val acc: {val['acc'] * 100:.1f}%")
        parts.append(f"train loss: {train['loss']:.4f} | val loss: {val['loss']:.4f}")
        parts += [f"val {k}: {v:.3f}" for k, v in val.items() if k not in ("loss", "acc")]
        if improved and save:
            parts.append("saved")
        if save and save_every_epoch:
            parts.append(f"kept epoch {epoch}")
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
