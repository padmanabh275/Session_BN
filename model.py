import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, dilation=dilation
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 Block - Regular Convolution
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),  # Output: 32x32
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),  # Output: 32x32
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),  # Output: 32x32
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.03),
        )
        
        # Skip connections
        self.skip1 = nn.Conv2d(48, 48, kernel_size=1)
        self.skip2 = nn.Conv2d(72, 72, kernel_size=1)
        
        # C2 Block - Depthwise Separable Convolution
        self.c2 = nn.Sequential(
            DepthwiseSeparableConv(48, 48, kernel_size=3, padding=2, dilation=2),  # Output: 32x32
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(48, 48, kernel_size=1),  # Output: 32x32
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        # C3 Block - Dilated Convolution
        self.c3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=3, dilation=3),  # Output: 32x32
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Conv2d(72, 72, kernel_size=1),  # Output: 32x32
            nn.BatchNorm2d(72),
            nn.ReLU(),
        )
        
        # C4 Block - Strided Convolution
        self.c4 = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, stride=2),  # Output: 16x16
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(96, 96, kernel_size=1),  # Output: 16x16
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(96, 10)

    def forward(self, x):
        # Forward with skip connections
        c1_out = self.c1(x)
        c2_out = self.c2(c1_out)
        c2_out = c2_out + self.skip1(c1_out)
        
        c3_out = self.c3(c2_out)
        c3_out = c3_out + self.skip2(c3_out)
        
        x = self.c4(c3_out)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x 