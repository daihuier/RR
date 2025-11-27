import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, t, _, _ = x.shape
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y


class FlowPhysNet(nn.Module):
    """
    光流版本 PhysNet（输入 2 通道：dx, dy）
    输出：波形 [B, T]
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(2, 16, (3,5,5), stride=(1,2,2), padding=(1,2,2))
        self.bn1   = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, 3, stride=(1,2,2), padding=1)
        self.bn2   = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm3d(64)

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.se   = SEBlock3D(64)

        self.dropout = nn.Dropout3d(0.3)
        self.wave_conv = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = self.res1(out)
        out = self.res2(out)
        out = self.se(out)
        out = self.dropout(out)

        B, _, T, _, _ = out.shape
        pooled = F.adaptive_avg_pool3d(out, (T,1,1))
        wave = self.wave_conv(pooled).view(B, T)
        return wave
