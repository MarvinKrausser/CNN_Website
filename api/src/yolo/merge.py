"""Combine COCO folders (images + <annotation_file>) into one COCO folder
that YoloDataset reads, e.g. WIDER and faces_scenery into one training set.

- Images are hard-linked into the output folder (copied if the folders are
  on different drives), named "<source>__<file_name>" so names never clash.
- Image and annotation ids are renumbered.
- Categories are merged by name and numbered 1, 2, ... in order of first
  appearance, so the same class in both sources becomes one class. --labels
  must equal the number of merged categories.
"""

import json
import os
import re
import shutil
from pathlib import Path

from src.yolo.config import Config


def source_names(folders):
    """Readable, unique prefix per source: data/wider_faces_coco/train -> wider_faces_coco_train."""
    names = []
    for folder in folders:
        name = re.sub(r"[^\w.-]+", "_", "_".join(folder.parts[-2:]))
        while name in names:
            name += "_"
        names.append(name)
    return names


def link_or_copy(source: Path, target: Path):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def merge_coco(cfg: Config):
    """Combine the COCO folders in --merge-dirs (comma separated) into --merge-out-dir."""
    folders = [Path(d.strip()) for d in cfg.merge_dirs.split(",") if d.strip()]
    if len(folders) < 2:
        raise SystemExit("Set --merge-dirs to at least two COCO folders, e.g. data/a/train,data/b/train")
    if not cfg.merge_out_dir:
        raise SystemExit("Set --merge-out-dir to the folder the combined dataset is written to")
    out = Path(cfg.merge_out_dir)
    if out.exists():
        raise SystemExit(f"{out} already exists, delete it or pick another --merge-out-dir")

    # read and check everything before writing anything
    sources = []
    for folder in folders:
        annotation_path = folder / cfg.annotation_file
        if not annotation_path.is_file():
            raise SystemExit(f"{annotation_path} not found")
        coco = json.loads(annotation_path.read_text())
        missing = [i["file_name"] for i in coco["images"] if not (folder / i["file_name"]).is_file()]
        if missing:
            raise SystemExit(f"{folder}: {len(missing)} images listed in {cfg.annotation_file} "
                             f"are missing, e.g. {missing[0]}")
        sources.append(coco)

    categories = {}  # name -> merged category
    images, annotations = [], []
    out.mkdir(parents=True)

    for folder, name, coco in zip(folders, source_names(folders), sources):
        category_ids = {}
        for category in coco["categories"]:
            merged = categories.setdefault(category["name"], {**category, "id": len(categories) + 1})
            category_ids[category["id"]] = merged["id"]

        image_ids = {}
        for image in coco["images"]:
            # flat output folder: keep subfolders of the source in the name
            file_name = f"{name}__{Path(image['file_name']).as_posix().replace('/', '__')}"
            link_or_copy(folder / image["file_name"], out / file_name)
            image_ids[image["id"]] = len(images) + 1
            images.append({**image, "id": len(images) + 1, "file_name": file_name})

        for annotation in coco["annotations"]:
            annotations.append({**annotation, "id": len(annotations) + 1,
                                "image_id": image_ids[annotation["image_id"]],
                                "category_id": category_ids[annotation["category_id"]]})

        print(f"{folder}: {len(coco['images'])} images, {len(coco['annotations'])} annotations "
              f"(prefix {name}__)")

    merged = {"images": images, "annotations": annotations, "categories": list(categories.values())}
    (out / cfg.annotation_file).write_text(json.dumps(merged))

    names = ", ".join(f"{c['id']} = {c['name']}" for c in merged["categories"])
    print(f"{out}: {len(images)} images, {len(annotations)} annotations, categories: {names}")
    print(f"Train with --labels {len(categories)}")
