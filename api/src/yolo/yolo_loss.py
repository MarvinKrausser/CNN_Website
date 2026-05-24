import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, predictions, targets, lambda_coord=5, lambda_noobj=0.5):
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_classes = predictions[..., 5:]
        target_boxes = targets[..., :4]
        target_conf = targets[..., 4]
        target_classes = targets[..., 5:]

        obj_mask = targets[..., 4] == 1
        noobj_mask = targets[..., 4] == 0


        if obj_mask.any():
            box_loss = F.mse_loss(
                pred_boxes[obj_mask],
                target_boxes[obj_mask],
                reduction="mean"
            )
        else:
            box_loss = torch.tensor(0.0, device=predictions.device)

        box_loss = lambda_coord * box_loss


        obj_loss = F.mse_loss(
            pred_conf[obj_mask],
            target_conf[obj_mask],
            reduction="mean"
        ) if obj_mask.any() else torch.tensor(0.0, device=predictions.device)

        noobj_loss = F.mse_loss(
            pred_conf[noobj_mask],
            target_conf[noobj_mask],
            reduction="mean"
        ) if noobj_mask.any() else torch.tensor(0.0, device=predictions.device)

        noobj_loss = lambda_noobj * noobj_loss


        class_loss = F.binary_cross_entropy_with_logits(
            pred_classes[obj_mask],
            target_classes[obj_mask]
        ) if obj_mask.any() else torch.tensor(0.0, device=predictions.device)

        
        total_loss = box_loss + obj_loss + noobj_loss + class_loss

        return total_loss