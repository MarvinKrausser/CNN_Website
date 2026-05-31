import torch.nn as nn
import torch.nn.functional as F

class  SeparableConvolution(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, kernel_size, groups=c_in, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x
    
class SkipBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return(F.relu(self.conv_skip(x) + self.conv(x), inplace=True))


class Bird_CNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),

            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            SkipBlock(c_in=c_hidden, c_out=c_hidden),
            
            SkipBlock(c_in=c_hidden, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),
            SkipBlock(c_in=c_hidden*2, c_out=c_hidden*2),

            nn.Conv2d(c_hidden*2, c_hidden*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden*4, c_out),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.model(x)