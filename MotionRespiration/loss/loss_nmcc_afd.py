# loss/loss_nmcc_afd.py
import torch
import torch.nn as nn


class NMCCAFDLoss(nn.Module):
    """
    NMCC + AFD + RR_L1 + 高频能量惩罚 的组合损失

    - NMCC: 频带内最大归一化互相关, 允许时移
            loss_nmcc = 1 - max_xcorr
    - AFD : 主峰频率差 (单位 bpm), 再按 AFD_SCALE 做归一 (不再 min(1, AFD/3) 截断)
    - RR_L1:
        用频带内“能量加权平均频率”得到软 RR, 再做 L1
        (比单纯 argmax 的梯度更平滑)
    - 高频谱惩罚:
        对高于 HIGH_BPM 的谱能量加一个小正则, 抑制假高频峰

    total = alpha * NMCC
           + (1 - alpha) * AFD
           + lambda_rr * RR_L1
           + lambda_high * HighFreqEnergy
    """

    def __init__(
        self,
        fps: float,
        min_bpm: float = 3.0,
        max_bpm: float = 42.0,
        alpha: float = 0.5,
        afd_scale: float = 10.0,
        lambda_rr: float = 0.3,
        lambda_high: float = 1e-4,
        high_bpm: float = 30.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.fps = float(fps)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.alpha = float(alpha)
        self.afd_scale = float(afd_scale)
        self.lambda_rr = float(lambda_rr)
        self.lambda_high = float(lambda_high)
        self.high_bpm = float(high_bpm)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred, gt: [B, T] 或 [B, 1, T]
        返回 dict:
            - total: 总 loss
            - nmcc : NMCC 分量
            - afd  : AFD 分量 (已经 / afd_scale)
            - rr_l1: RR_L1 分量 (已经 / afd_scale)
            - high_energy: 高频能量正则
            - max_corr_mean: 最大互相关均值 (监控)
            - afd_bpm_mean : 平均 |ΔBPM|
            - rr_fft_pred_mean / rr_fft_gt_mean: 软 RR 均值 (监控)
        """
        # ----------- 对齐形状 -----------
        if pred.ndim == 3:
            pred = pred.squeeze(1)  # [B, T]
        if gt.ndim == 3:
            gt = gt.squeeze(1)

        B, T = pred.shape
        device = pred.device

        # ----------- 去均值 -----------
        pred = pred - pred.mean(dim=-1, keepdim=True)
        gt = gt - gt.mean(dim=-1, keepdim=True)

        # ----------- FFT & 频带选择 -----------
        spec_p = torch.fft.rfft(pred, dim=-1)  # [B, F]
        spec_g = torch.fft.rfft(gt, dim=-1)    # [B, F]

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(device)  # Hz, [F]
        bpm = freqs * 60.0                                          # [F]

        band_mask = (bpm >= self.min_bpm) & (bpm <= self.max_bpm)
        if not band_mask.any():
            # clip 太短之类的极端情况: 不做 band 限制
            band_mask[:] = True

        # 频带内频率向量 [F_band]
        bpm_band = bpm[band_mask]

        # ----------- 用 band_mask 做带通 (NMCC 仍然用 full-length 时域) -----------
        spec_p_band = spec_p.clone()
        spec_g_band = spec_g.clone()
        spec_p_band[..., ~band_mask] = 0
        spec_g_band[..., ~band_mask] = 0

        # ----------- 带通后的时域信号 (用于 NMCC) -----------
        pred_bp = torch.fft.irfft(spec_p_band, n=T, dim=-1)  # [B, T]
        gt_bp = torch.fft.irfft(spec_g_band, n=T, dim=-1)

        norm_p = torch.sqrt((pred_bp ** 2).sum(dim=-1) + self.eps)  # [B]
        norm_g = torch.sqrt((gt_bp ** 2).sum(dim=-1) + self.eps)    # [B]

        # ----------- 频带内互相关 + 归一 -----------
        cross_spec = spec_p_band * torch.conj(spec_g_band)          # [B, F]
        xcorr = torch.fft.irfft(cross_spec, n=T, dim=-1)            # [B, T]

        denom = (norm_p * norm_g).unsqueeze(-1) + self.eps          # [B,1]
        xcorr_norm = xcorr / denom                                  # [B,T]

        max_corr, _ = xcorr_norm.max(dim=-1)                        # [B]
        nmcc_each = 1.0 - max_corr                                  # [B]
        nmcc_loss = nmcc_each.mean()

        # ----------- AFD: 主峰频率差 (单位 bpm) -----------
        mag_p_band = spec_p[..., band_mask].abs()                   # [B, F_band]
        mag_g_band = spec_g[..., band_mask].abs()                   # [B, F_band]

        peak_idx_p = mag_p_band.argmax(dim=-1)                      # [B]
        peak_idx_g = mag_g_band.argmax(dim=-1)                      # [B]

        peak_bpm_p = bpm_band[peak_idx_p]                           # [B]
        peak_bpm_g = bpm_band[peak_idx_g]                           # [B]

        afd_bpm = (peak_bpm_p - peak_bpm_g).abs()                   # [B]
        afd_loss = (afd_bpm / self.afd_scale).mean()

        # ----------- RR_L1: 频谱加权平均 RR 差 -----------
        weights_p = mag_p_band + self.eps                           # [B,F_band]
        weights_g = mag_g_band + self.eps

        weights_p = weights_p / weights_p.sum(dim=-1, keepdim=True)
        weights_g = weights_g / weights_g.sum(dim=-1, keepdim=True)

        rr_fft_p = (weights_p * bpm_band.unsqueeze(0)).sum(dim=-1)  # [B]
        rr_fft_g = (weights_g * bpm_band.unsqueeze(0)).sum(dim=-1)  # [B]

        rr_l1 = ((rr_fft_p - rr_fft_g).abs() / self.afd_scale).mean()

        # ----------- 高频谱能量惩罚 -----------
        high_mask = bpm >= self.high_bpm
        if high_mask.any():
            mag_p_full = spec_p.abs()                               # [B,F]
            high_energy = (mag_p_full[..., high_mask] ** 2).mean()
        else:
            high_energy = torch.tensor(0.0, device=device)

        # ----------- 组合总 loss -----------
        total_loss = (
            self.alpha * nmcc_loss
            + (1.0 - self.alpha) * afd_loss
            + self.lambda_rr * rr_l1
            + self.lambda_high * high_energy
        )

        return {
            "total": total_loss,
            "nmcc": nmcc_loss,
            "afd": afd_loss,
            "rr_l1": rr_l1,
            "high_energy": high_energy,
            "max_corr_mean": max_corr.mean(),
            "afd_bpm_mean": afd_bpm.mean(),
            "rr_fft_pred_mean": rr_fft_p.mean(),
            "rr_fft_gt_mean": rr_fft_g.mean(),
        }
