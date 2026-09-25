"""Bounding box helpers. Boxes are [xmin, ymin, xmax, ymax] unless noted.

Used by training and by the production server, so keep this free of
training-only dependencies."""


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxA_area + boxB_area - inter_area

    return inter_area / union if union > 0 else 0


def nms(bboxes, iou_threshold=0.1):
    """bboxes: [confidence, xmin, ymin, xmax, ymax]. Returns boxes without
    the confidence, highest confidence first."""
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

    keep = []
    while bboxes:
        best = bboxes.pop(0)
        keep.append(best[1:5])
        bboxes = [box for box in bboxes if iou(best[1:5], box[1:5]) < iou_threshold]

    return keep


def xy_center_to_edges(xcenter, ycenter, width, height):
    width = max(width, 1)
    height = max(height, 1)

    x = xcenter - (width / 2)
    y = ycenter - (height / 2)

    return [x, y, x + width, y + height]


def cell_to_image_coords(x, y, img_w, img_h, S, cell_i, cell_j):
    """Grid-cell-relative position -> image position."""
    return x + (img_w / S) * cell_i, y + (img_h / S) * cell_j


def image_to_cell_coords(x, y, img_w, img_h, S, cell_i, cell_j):
    """Image position -> grid-cell-relative position."""
    return x - (img_w / S) * cell_i, y - (img_h / S) * cell_j


def is_center_in_grid_cell(x, y, img_w, img_h, S, cell_i, cell_j):
    gt_cell_i = min(int(x / (img_w / S)), S - 1)
    gt_cell_j = min(int(y / (img_h / S)), S - 1)
    return (gt_cell_i == cell_i) and (gt_cell_j == cell_j)


def scale_box(box, scale_w, scale_h):
    return [int(box[0] * scale_w), int(box[1] * scale_h), int(box[2] * scale_w), int(box[3] * scale_h)]
