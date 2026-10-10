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

    # convert-wider: WIDER FACE -> COCO folders <wider_out_dir>/train and /test
    wider_dir: str = ""               # contains WIDER_train/, WIDER_val/, wider_face_split/
    wider_out_dir: str = "data/wider_face"
    wider_min_face: float = 2.0       # drop smaller faces (px after resizing to image_size)

    # Model
    c_hidden: int = 16
    boxes: int = 1
    labels: int = 1
    size_activation: str = "sigmoid"  # "sigmoid" (current training) or "exp" (deployed model)

    # Training
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3       # peak AdamW learning rate
    weight_decay: float = 0.05        # AdamW, conv weights only (not biases / BatchNorm)
    warmup_epochs: int = 3            # linear warmup from learning_rate / 10
    min_learning_rate: float = 1e-5   # cosine decay ends here at the last epoch
    train_workers: int = 2
    val_workers: int = 0
    device: str = "auto"              # "auto", "cpu", "cuda:0", ...

    # Output
    save_dir: str = "saved_models"
    model_name: str = "face_detection_yolo"
    save: bool = True

    # Run log (src/common/run_log.py): one folder per training run
    runs_dir: str = "runs"            # empty = no run log
    run_name: str = ""                # added to the run folder name, e.g. "bn-lr3e-4"

    # sample / webcam tasks
    checkpoint: str = ""              # empty = <save_dir>/<model_name>/<model_name>
    threshold: float = 0.95           # confidence needed to draw a box
    camera: int = 0                   # webcam index
