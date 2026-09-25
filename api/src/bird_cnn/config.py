from dataclasses import dataclass


@dataclass
class Config:
    """Settings for the bird classifier. Paths are relative to api/.
    Every field can be overridden on the command line, e.g. --epochs 20."""

    # Data
    data_dir: str = "data/CUB_200_2011/images"   # ImageFolder layout: one folder per class
    train_fraction: float = 0.8
    image_size: int = 128

    # Model
    num_classes: int = 200
    c_hidden: int = 16

    # Training
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"                          # "auto", "cpu", "cuda:0", ...

    # Output
    save_dir: str = "saved_models"
    model_name: str = "bird_cnn"
    save: bool = True

    # predict task
    image: str = "test1.jpg"
    checkpoint: str = ""                          # empty = <save_dir>/<model_name>/<model_name>
