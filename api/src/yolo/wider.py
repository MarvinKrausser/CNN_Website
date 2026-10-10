"""Convert the WIDER FACE dataset to the COCO folders YoloDataset reads.

WIDER_train becomes <wider_out_dir>/train and WIDER_val becomes
<wider_out_dir>/test (WIDER's own test split has no public boxes). Each folder
gets its images (copied, flat) and <annotation_file> with one category,
1 = face, like faces_scenery.

Most WIDER faces are tiny faces in crowds, which the model cannot use, so:
- faces smaller than --wider-min-face px after resizing to --image-size are dropped,
- images where two faces fall in the same --grid cell are skipped. YoloDataset
  keeps one face per cell, so the other face would be trained as background.
"""

import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from src.common.boxes import grid_cell
from src.yolo.config import Config

# WIDER split -> output folder (train_dir / val_dir)
SPLITS = {"train": "train", "val": "test"}


def find(root: Path, pattern: str) -> Path:
    """First match of pattern below root, so nested unzips (WIDER_train/WIDER_train/...) work too."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return matches[0]


def parse_ground_truth(path: Path):
    """Yield (image path, faces), one face per row as
    [x, y, w, h, blur, expression, illumination, invalid, occlusion, pose]."""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        name, count = lines[i], int(lines[i + 1])
        faces = [[int(v) for v in line.split()] for line in lines[i + 2 : i + 2 + count]]
        i += 2 + max(count, 1)  # an image without faces still has one "0 0 0 ..." row
        yield name, faces


def usable_boxes(faces, img_w, img_h, image_size, min_face):
    """WIDER faces -> COCO [xmin, ymin, w, h] boxes in whole pixels, clipped to
    the image and at least min_face px wide and high once resized to image_size."""
    boxes = []
    for x, y, w, h, _blur, _expression, _illumination, invalid, _occlusion, _pose in faces:
        x0, y0 = max(x, 0), max(y, 0)
        w, h = min(x + w, img_w) - x0, min(y + h, img_h) - y0
        if invalid or w < 1 or h < 1:
            continue
        if w * image_size / img_w < min_face or h * image_size / img_h < min_face:
            continue
        boxes.append([x0, y0, w, h])
    return boxes


def shares_grid_cell(boxes, img_w, img_h, image_size, grid, eps=1e-3):
    """True when two box centres can land in the same grid cell after resizing
    to image_size. A centre within eps px of a cell border counts for both
    cells, since Albumentations' float32 math can round it to either side."""
    sx, sy = image_size / img_w, image_size / img_h
    seen = set()
    for x, y, w, h in boxes:
        cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
        cells = {grid_cell(cx + dx, cy + dy, image_size, image_size, grid)
                 for dx in (-eps, eps) for dy in (-eps, eps)}
        if cells & seen:
            return True
        seen |= cells
    return False


def convert_split(images_dir: Path, gt_file: Path, out_dir: Path, cfg: Config):
    out_dir.mkdir(parents=True)
    images, annotations = [], []
    skipped = {"unreadable": 0, "faces sharing a cell": 0}

    for name, faces in tqdm(list(parse_ground_truth(gt_file)), desc=out_dir.name):
        source = images_dir / name
        # read with OpenCV like YoloDataset does, so every kept image is loadable
        image = cv2.imread(str(source))
        if image is None:
            skipped["unreadable"] += 1
            continue
        img_h, img_w = image.shape[:2]

        boxes = usable_boxes(faces, img_w, img_h, cfg.image_size, cfg.wider_min_face)
        if shares_grid_cell(boxes, img_w, img_h, cfg.image_size, cfg.grid):
            skipped["faces sharing a cell"] += 1
            continue

        target = out_dir / source.name
        if target.exists():
            raise FileExistsError(f"two WIDER images are named {source.name}")
        shutil.copy2(source, target)

        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": source.name, "width": img_w, "height": img_h})
        for box in boxes:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "bbox": box, "area": box[2] * box[3], "iscrowd": 0})

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "face", "supercategory": ""}]}
    (out_dir / cfg.annotation_file).write_text(json.dumps(coco))

    empty = len(images) - len({a["image_id"] for a in annotations})
    print(f"{out_dir}: {len(images)} images ({empty} without faces), {len(annotations)} faces. "
          f"Skipped: " + ", ".join(f"{n} {reason}" for reason, n in skipped.items()))


def convert_wider(cfg: Config):
    """Convert WIDER FACE (--wider-dir) to COCO train/ and test/ folders in --wider-out-dir."""
    if not cfg.wider_dir:
        raise SystemExit("Set --wider-dir to the folder with WIDER_train/, WIDER_val/ and wider_face_split/")
    root, out = Path(cfg.wider_dir), Path(cfg.wider_out_dir)

    sources = {split: (find(root, f"WIDER_{split}/images"), find(root, f"wider_face_{split}_bbx_gt.txt"))
               for split in SPLITS}
    for folder in SPLITS.values():
        if (out / folder).exists():
            raise SystemExit(f"{out / folder} already exists, delete it or pick another --wider-out-dir")

    for split, folder in SPLITS.items():
        images_dir, gt_file = sources[split]
        convert_split(images_dir, gt_file, out / folder, cfg)

    print(f"Train with: --train-dir {out / 'train'} --val-dir {out / 'test'} "
          f"--image-size {cfg.image_size} --grid {cfg.grid} --labels 1")
