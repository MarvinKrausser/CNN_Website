import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_prediction(label, image, threshold=0.9):
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
            if label[x, y, 4].item() < threshold:
                grids_to_draw_noobj.append([x*grid_size, y*grid_size, (x+1)*grid_size, (y+1)*grid_size]) #xmin, ymin, xmax, ymax
                continue
            grids_to_draw_obj.append([x*grid_size, y*grid_size, (x+1)*grid_size, (y+1)*grid_size]) #xmin, ymin, xmax, ymax

            boxx = label[x, y, 0] * (image_size / grid_number)
            boxy = label[x, y, 1] * (image_size / grid_number)

            boxw = label[x, y, 2] * image_size
            boxh = label[x, y, 3] * image_size

            boxx, boxy = turn_image_centered(x=boxx, y=boxy, img_w=image_size, img_h=image_size, S=grid_number, cell_i=x, cell_j=y)

            boxes_to_draw.append([label[x, y, 4]] + xy_center_to_edges(boxx, boxy, boxw, boxh)) #xmin, ymin, xmax, ymax
    boxes_to_draw = nms(boxes_to_draw)
    return boxes_to_draw, grids_to_draw_obj, grids_to_draw_noobj

def turn_image_centered(x, y, img_w, img_h, S, cell_i, cell_j):
    cell_w = img_w / S
    cell_h = img_h / S

    cell_border_w = cell_w * cell_i
    cell_border_h = cell_h * cell_j

    return x + cell_border_w, y + cell_border_h

def xy_center_to_edges(xcenter, ycenter, width, height):
    width = max(width, 1)
    height = max(height, 1)

    x = xcenter - (width / 2)
    y = ycenter - (height / 2)

    return [x, y, x + width, y + height]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxB_area = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = boxA_area + boxB_area - inter_area

    return inter_area / union if union > 0 else 0

def nms(bboxes, iou_threshold=0.1):
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

    keep = []

    while bboxes:
        best = bboxes.pop(0)
        keep.append(best[1:5])

        bboxes = [
            box for box in bboxes
            if iou(best[1:5], box[1:5]) < iou_threshold
        ]

    return keep

class Yolo_Conv_Block(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, kernel_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_hidden, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_hidden, out_channels=c_out, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)
    
class SkipBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(num_groups=c_out//8, num_channels=c_out),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(num_groups=c_out//8, num_channels=c_out),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(num_groups=c_out//8, num_channels=c_out),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1),
            nn.GroupNorm(num_groups=c_out//8, num_channels=c_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return(F.dropout(F.leaky_relu(self.conv_skip(x) + self.conv(x), inplace=True), p=0.3))


class Yolo_model(nn.Module):
    def __init__(self, c_in, boxes, grid, labels, c_hidden=16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=c_hidden//8, num_channels=c_hidden),
            nn.LeakyReLU(inplace=True),

            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            
            SkipBlock(c_in=c_hidden, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),

            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*4),
            SkipBlock(c_in=c_hidden*4, c_out=c_hidden*4),
            SkipBlock(c_in=c_hidden*4, c_out=c_hidden*4),
            SkipBlock(c_in=c_hidden*4, c_out=c_hidden*4),

            nn.Conv2d(c_hidden*4, c_hidden*8, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=c_hidden//2, num_channels=c_hidden*8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((grid, grid)),
            nn.Conv2d(c_hidden*8, boxes*5 + labels, kernel_size=1)
        )

    def forward(self, x):
        x = self.model(x).permute(0, 2, 3, 1)
        center = F.sigmoid(x[..., :2])
        size = torch.exp(x[..., 2:4])
        conf_class = F.sigmoid(x[..., 4:])
        return torch.cat([center, size, conf_class], dim=3)