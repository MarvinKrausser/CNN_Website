import torch.nn as nn


class ObjectDetectionCNN(nn.Module):
    """Small crop classifier used by the R-CNN style experiments
    (selective search proposals -> classify each crop)."""

    def __init__(self, c_in, c_hidden, c_out, layers):
        super().__init__()

        self.model = nn.ModuleList()

        self.model.append(nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True)
        ))

        for _ in range(layers - 1):
            self.model.append(nn.Sequential(
                nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_hidden),
                nn.ReLU(inplace=True)
            ))

        self.model.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden, c_out),
            nn.Dropout(0.3)
        ))

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
