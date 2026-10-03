"""Detection metrics over the whole validation set, logged next to the loss.
Per grid cell, like the loss: a cell is a detection when its confidence
passes `conf_threshold`, and a correct one when the cell holds a face and
the box overlaps the true box by at least `iou_threshold`."""

import torch


def _cell_boxes(t):
    """[N, S, S, >=4] cell outputs (x, y relative to the cell, w, h relative to
    the image) -> [N, S, S, 4] boxes as (xmin, ymin, xmax, ymax) in image units."""
    S = t.shape[1]
    i = torch.arange(S, device=t.device).view(S, 1)
    j = torch.arange(S, device=t.device).view(1, S)
    cx = (i + t[..., 0]) / S
    cy = (j + t[..., 1]) / S
    w, h = t[..., 2], t[..., 3]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _iou(a, b):
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def _average_precision(scores, labels):
    """How well the confidence ranks cells with a face above empty cells (1 = perfect)."""
    order = scores.argsort(descending=True)
    hits = labels[order].float()
    precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1, device=hits.device)
    return ((precision * hits).sum() / hits.sum().clamp(min=1)).item()


def detection_metrics(preds, targets, conf_threshold=0.5, iou_threshold=0.5):
    obj = targets[..., 4] == 1
    conf = preds[..., 4]
    ious = _iou(_cell_boxes(preds), _cell_boxes(targets))

    detected = conf > conf_threshold
    correct = (detected & obj & (ious > iou_threshold)).sum().item()
    n_obj = obj.sum().item()
    return {
        "conf_ap": _average_precision(conf.flatten(), obj.flatten()),
        "box_iou": ious[obj].mean().item() if n_obj else 0.0,
        "precision": correct / max(detected.sum().item(), 1),
        "recall": correct / max(n_obj, 1),
    }
