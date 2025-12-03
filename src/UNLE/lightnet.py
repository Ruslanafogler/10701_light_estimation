# UNLE/lightnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightNet(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # x: (B, 5, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)           # (B, 128, 4, 4)
        x = x.view(x.size(0), -1)  # (B, 128*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # (B, 3)
        return x
