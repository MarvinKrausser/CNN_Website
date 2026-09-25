from dataclasses import dataclass


@dataclass
class Config:
    """Settings for the R-CNN style experiments (classify crops, then run
    selective search on a full image). Paths are relative to api/.
    Every field can be overridden on the command line, e.g. --epochs 20."""

    # Datasets
    faces_raw_dir: str = "data/faces/train"             # COCO: source for create-dataset
    faces_processed_dir: str = "data/faces_processed"   # ImageFolder: face/ and background/
    football_train_dir: str = "data/football/train"     # COCO
    football_val_dir: str = "data/football/valid"       # COCO
    annotation_file: str = "_annotations.coco.json"

    # Crops
    crop_size: int = 64
    padding: int = 40                     # football: extra pixels around each box/proposal
    backgrounds_per_image: int = 10       # create-dataset: background crops per image
    train_fraction: float = 0.8

    # Model
    c_hidden: int = 32
    layers: int = 10
    num_classes: int = 2

    # Training
    epochs: int = 200
    batch_size: int = 256                 # faces (pre-cropped images)
    football_batch_size: int = 2          # football (full images, cropped on the fly)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    background_weight: float = 1 / 3      # class weight of "background" in the faces loss
    device: str = "auto"

    # Output
    save_dir: str = "saved_models"
    faces_model_name: str = "face_detection"
    football_model_name: str = "object_detection"
    save: bool = True

    # detect tasks: proposals smaller than min_size or larger than max_size are ignored
    image: str = ""                       # detect-faces: image to run on
    min_size: int = 5
    max_size: int = 600
    min_confidence: float = 0.97
    football_max_size: int = 100
    football_min_confidence: float = 0.8
