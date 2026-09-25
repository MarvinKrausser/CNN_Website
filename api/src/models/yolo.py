"""YOLO-style single-shot face detector. Shared by training and the
production server; the layer names must not change or saved weights stop
loading."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.boxes import cell_to_image_coords, nms, xy_center_to_edges


class SkipBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=c_out // 8, num_channels=c_out),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=c_out // 8, num_channels=c_out),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups=c_out // 8, num_channels=c_out),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1),
            nn.GroupNorm(num_groups=c_out // 8, num_channels=c_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # F.dropout defaults to training=True, so this stays active in eval
        # mode too. Kept as-is because the saved models were trained this way.
        return F.dropout(F.leaky_relu(self.conv_skip(x) + self.conv(x), inplace=True), p=0.3)


class Yolo_model(nn.Module):
    """Output: [batch, grid, grid, boxes*5 + labels] with per cell
    (x, y, w, h, confidence, classes...).

    size_activation decides how w/h are produced:
      "exp"     - exp(x), used by the deployed build_models/face_detection_yolo
      "sigmoid" - sigmoid(x), what the training code switched to later
    """

    def __init__(self, c_in, boxes, grid, labels, c_hidden=16, size_activation="sigmoid"):
        super().__init__()
        if size_activation not in ("exp", "sigmoid"):
            raise ValueError(f"size_activation must be 'exp' or 'sigmoid', got {size_activation!r}")
        self.size_activation = size_activation

        self.model = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=c_hidden // 8, num_channels=c_hidden),
            nn.LeakyReLU(inplace=True),

            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),

            SkipBlock(c_in=c_hidden, c_out=c_hidden * 2),
            SkipBlock(c_in=c_hidden * 2, c_out=c_hidden * 2),
            SkipBlock(c_in=c_hidden * 2, c_out=c_hidden * 2),
            SkipBlock(c_in=c_hidden * 2, c_out=c_hidden * 2),

            SkipBlock(c_in=c_hidden * 2, c_out=c_hidden * 4),
            SkipBlock(c_in=c_hidden * 4, c_out=c_hidden * 4),
            SkipBlock(c_in=c_hidden * 4, c_out=c_hidden * 4),
            SkipBlock(c_in=c_hidden * 4, c_out=c_hidden * 4),

            nn.Conv2d(c_hidden * 4, c_hidden * 8, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=c_hidden // 2, num_channels=c_hidden * 8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((grid, grid)),
            nn.Conv2d(c_hidden * 8, boxes * 5 + labels, kernel_size=1)
        )

    def forward(self, x):
        x = self.model(x).permute(0, 2, 3, 1)
        if self.size_activation == "sigmoid":
            return F.sigmoid(x)
        center = F.sigmoid(x[..., :2])
        size = torch.exp(x[..., 2:4])
        conf_class = F.sigmoid(x[..., 4:])
        return torch.cat([center, size, conf_class], dim=3)


def convert_prediction(label, image, threshold=0.9):
    """Turns one model output (or target) [grid, grid, 5 + classes] into boxes
    in image pixels. Returns (boxes after NMS, cells with an object, cells
    without one); the cell lists are only used for visualisation."""
    image = image.clone().detach()
    label = label.clone().detach()

    image_size = image.shape[1]
    grid_number = label.shape[0]
    grid_size = image_size / grid_number

    boxes_to_draw = []
    grids_to_draw_obj = []
    grids_to_draw_noobj = []
    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            cell = [x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size]
            if label[x, y, 4].item() < threshold:
                grids_to_draw_noobj.append(cell)
                continue
            grids_to_draw_obj.append(cell)

            boxx = label[x, y, 0] * (image_size / grid_number)
            boxy = label[x, y, 1] * (image_size / grid_number)

            boxw = label[x, y, 2] * image_size
            boxh = label[x, y, 3] * image_size

            boxx, boxy = cell_to_image_coords(x=boxx, y=boxy, img_w=image_size, img_h=image_size,
                                              S=grid_number, cell_i=x, cell_j=y)

            boxes_to_draw.append([label[x, y, 4]] + xy_center_to_edges(boxx, boxy, boxw, boxh))

    boxes_to_draw = nms(boxes_to_draw)
    return boxes_to_draw, grids_to_draw_obj, grids_to_draw_noobj
