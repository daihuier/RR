import torch
import torch.nn as nn
from model.physnet_model import PhysNet
from model.flow_physnet import FlowPhysNet


class PhysNetFusion(nn.Module):
    """
    RGB + Flow 双分支 → 波形 Late Fusion
    """
    def __init__(self):
        super().__init__()
        self.rgb_net  = PhysNet()        # 原 PhysNet
        self.flow_net = FlowPhysNet()    # 光流版

        # 融合：简单 1×1 线性层
        self.fuse = nn.Conv1d(2, 1, kernel_size=1)

    def forward(self, rgb_clip, flow_clip):
        # rgb_clip  : [B,3,T,H,W]
        # flow_clip : [B,2,T,H,W]

        wave_rgb  = self.rgb_net(rgb_clip)     # [B,T]
        wave_flow = self.flow_net(flow_clip)   # [B,T]

        x = torch.stack([wave_rgb, wave_flow], dim=1)  # [B,2,T]
        fused = self.fuse(x).squeeze(1)                # [B,T]

        return fused
