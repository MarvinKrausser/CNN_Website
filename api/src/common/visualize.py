"""Plot helpers for local experiments (matplotlib is not installed in the
production image, so it is imported lazily)."""


def show_image(image, title=None):
    """image: tensor [C, H, W]."""
    from matplotlib import pyplot as plt

    plt.figure(figsize=(3, 3))
    plt.imshow(image.permute(1, 2, 0))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_batch(loader, count=4):
    from matplotlib import pyplot as plt

    images, labels = next(iter(loader))
    for i in range(min(count, len(images))):
        plt.figure(figsize=(3, 3))
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()
