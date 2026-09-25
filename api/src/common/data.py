import torch
from torch.utils.data import Dataset, random_split


class TransformedSubset(Dataset):
    """Applies a transform to a Subset, so train and validation splits of the
    same dataset can use different transforms."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def split_dataset(dataset, train_fraction: float, train_transform, val_transform):
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return (
        TransformedSubset(train_subset, train_transform),
        TransformedSubset(val_subset, val_transform),
    )


def collate_as_tuples(batch):
    """For detection datasets whose targets are dicts of different sizes."""
    return tuple(zip(*batch))
