from dataclasses import dataclass


@dataclass
class Config:
    """Settings for the YOLO face detector. Paths are relative to api/.
    Every field can be overridden on the command line, e.g. --batch-size 32."""

    # Data (COCO format: images + _annotations.coco.json in each folder)
    train_dir: str = "data/faces_scenery/train"
    val_dir: str = "data/faces_scenery/test"
    annotation_file: str = "_annotations.coco.json"
    augment_train: bool = True
    augment_val: bool = True          # the original code augmented validation too
    image_size: int = 64
    grid: int = 6

    # Model
    c_hidden: int = 16
    boxes: int = 1
    labels: int = 1
    size_activation: str = "sigmoid"  # "sigmoid" (current training) or "exp" (deployed model)

    # Training
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_workers: int = 2
    val_workers: int = 0
    device: str = "auto"              # "auto", "cpu", "cuda:0", ...

    # Output
    save_dir: str = "saved_models"
    model_name: str = "face_detection_yolo"
    save: bool = True

    # sample / webcam tasks
    checkpoint: str = ""              # empty = <save_dir>/<model_name>/<model_name>
    threshold: float = 0.95           # confidence needed to draw a box
    camera: int = 0                   # webcam index
