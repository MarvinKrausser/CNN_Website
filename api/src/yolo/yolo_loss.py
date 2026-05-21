import torch
import torch.nn as nn

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
        
        box_loss = lambda_coord * torch.mean((pred_boxes - target_boxes) ** 2)

        obj_loss = torch.mean((pred_conf[target_conf == 1] - target_conf[target_conf == 1]) ** 2)
        noobj_loss = lambda_noobj * torch.mean((pred_conf[target_conf == 0]) ** 2)

        class_loss = torch.mean((pred_classes[target_conf == 1] - target_classes[target_conf == 1]) ** 2)

        total_loss = box_loss + obj_loss + noobj_loss + class_loss
        return total_loss
