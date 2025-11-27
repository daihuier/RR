# model/physnet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y


class PhysNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 5, 5),
            stride=(1, 2, 2),
            padding=(1, 2, 2),
        )
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.bn3 = nn.BatchNorm3d(64)

        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.se = SEBlock3D(64, reduction=8)

        self.dropout = nn.Dropout3d(p=0.3)
        self.waveform_conv = nn.Conv3d(64, 1, kernel_size=(1, 1, 1))

    def extract_feature(self, x: torch.Tensor):
        B, C, T, H, W = x.shape

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.se(out)
        out = self.dropout(out)

        return out, B, T


class PhysNet(PhysNetBase):
    """
    只预测呼吸波形的 PhysNet：
    输入: x [B,3,T,H,W]
    输出: pred_waveform [B,T]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, B, T = self.extract_feature(x)
        feat_pool = F.adaptive_avg_pool3d(feat, (T, 1, 1))
        waveform_feat = self.waveform_conv(feat_pool)
        pred_waveform = waveform_feat.view(B, T)
        return pred_waveform


class PhysNetWithRR(PhysNet):
    """
    兼容旧代码的壳：
    - 继续用 PhysNet 预测波形
    - 额外返回一个 RR 标量（这里随便给个平均值，当占位用）
    旧脚本如果还在用 (wave, rr) 这个接口，就不会炸。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        wave = super().forward(x)
        rr = wave.abs().mean(dim=1)
        return wave, rr
