import torch


def get_device(name: str = "auto") -> torch.device:
    """"auto" picks CUDA when available, otherwise CPU."""
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    print("Using device", device)
    return device
