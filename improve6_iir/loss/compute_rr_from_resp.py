"""
loss/compute_rr_from_resp.py

统一的呼吸频率估计函数，用于从时间序列呼吸信号中估计 RR（bpm），并支持频域带通。
相对旧版本，这一版的改动点：
- 引入 upsample_factor，通过 FFT 零填充细化频率网格，从而把 RR 分辨率从 3 bpm 级别提升到 < 1 bpm；
- 在主峰附近做简单抛物线插值，进一步提升峰值位置估计精度；
- 所有入口保持兼容：compute_rr_from_resp(x, fs, rr_min, rr_max) 的调用方式不需要改。
"""

from typing import Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ArrayLike = Union[np.ndarray, "torch.Tensor", list, tuple]


def _to_numpy_1d(x: ArrayLike) -> np.ndarray:
    """
    把输入转成 numpy 1D 向量，并清掉 NaN / Inf。
    """
    if HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    # 去掉非有限值
    mask = np.isfinite(x)
    if mask.sum() < 4:
        return x.astype(np.float64)
    x = x[mask]
    return x.astype(np.float64)


def _fallback_rr(rr_min: float, rr_max: float) -> float:
    """
    出现异常时的兜底 RR，取频段中心。
    """
    return float(0.5 * (rr_min + rr_max))


def compute_rr_from_resp(
    x: ArrayLike,
    fs: float,
    rr_min: float = 6.0,
    rr_max: float = 25.0,
    upsample_factor: int = 4,
) -> float:
    """
    从 1D 呼吸波形估计呼吸频率 RR（bpm）。

    参数
    ----
    x : 1D 序列（numpy / torch / list 都可以）
    fs: 采样率，Hz（你这里是 30.0）
    rr_min, rr_max: 期望的 RR 搜索范围（bpm）
    upsample_factor: 频域插值因子，通过 rFFT 的 n 参数做零填充，
                     比如 T=600, fs=30：
                        - 不插值：分辨率 60*fs/T = 3 bpm
                        - upsample_factor=4：分辨率 0.75 bpm

    返回
    ----
    rr_bpm: 估计的呼吸频率，单位 bpm（float）
    """
    try:
        sig = _to_numpy_1d(x)
        n = sig.shape[0]
        if n < 8 or fs <= 0:
            return _fallback_rr(rr_min, rr_max)

        # 去均值
        sig = sig - sig.mean()

        # Hann 窗，减小频谱泄漏
        win = np.hanning(n)
        sig_win = sig * win

        # -------- 频域分析（零填充 + 截频段）--------
        up = max(1, int(upsample_factor))
        n_fft = int(n * up)

        spec = np.fft.rfft(sig_win, n=n_fft)          # 复数频谱
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)    # Hz
        bpm_grid = freqs * 60.0                       # bpm

        band_mask = (bpm_grid >= rr_min) & (bpm_grid <= rr_max)
        if not np.any(band_mask):
            return _fallback_rr(rr_min, rr_max)

        mag = np.abs(spec[band_mask])
        bpm_band = bpm_grid[band_mask]                # 这里的步长已经被 up 放大细化

        # 至少要有 3 个点才能做简单抛物线插值
        if mag.size < 3:
            # 只有 1~2 个点，就直接取最大值位置
            idx = int(np.argmax(mag))
            rr_peak = float(bpm_band[idx])
            return float(np.clip(rr_peak, rr_min, rr_max))

        # 主峰所在的索引（在带通频带内的局部索引）
        k = int(np.argmax(mag))

        # -------- 抛物线插值（parabolic interpolation）--------
        # 在 k-1, k, k+1 三个幅度点上拟合一条抛物线，求其顶点位置相对 k 的偏移 Δ
        if 1 <= k <= mag.size - 2:
            y0, y1, y2 = mag[k - 1], mag[k], mag[k + 1]
            denom = (y0 - 2.0 * y1 + y2)
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom      # Δ 在 [-0.5, 0.5] 附近
            else:
                delta = 0.0
        else:
            # 峰值在边界上，没法做三点插值
            delta = 0.0

        # 频率网格的步长（bpm）
        # band 内步长基本是常数，直接用相邻点差即可
        if mag.size >= 2:
            bpm_step = float(bpm_band[1] - bpm_band[0])
        else:
            bpm_step = 60.0 * fs / n_fft  # 理论值，兜底

        rr_peak = float(bpm_band[k] + delta * bpm_step)
        rr_peak = float(np.clip(rr_peak, rr_min, rr_max))
        if not np.isfinite(rr_peak):
            return _fallback_rr(rr_min, rr_max)
        return rr_peak

    except Exception:
        return _fallback_rr(rr_min, rr_max)


def bandpass_only(
    x: ArrayLike,
    fs: float,
    rr_min: float = 6.0,
    rr_max: float = 25.0,
    upsample_factor: int = 1,
) -> np.ndarray:
    """
    一个简单的“频域带通 + 反变换”工具：
    - 用于把呼吸波形的频谱限制在 [rr_min, rr_max] 之间；
    - 不依赖 scipy，只用 numpy 的 FFT；
    - 主要给数据预处理/可视化用，对训练流程没有硬依赖。

    返回值是和输入等长的 numpy 1D 向量。
    """
    sig = _to_numpy_1d(x)
    n = sig.shape[0]
    if n < 8 or fs <= 0:
        return sig

    sig = sig - sig.mean()
    win = np.hanning(n)
    sig_win = sig * win

    up = max(1, int(upsample_factor))
    n_fft = int(n * up)

    spec = np.fft.rfft(sig_win, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    bpm_grid = freqs * 60.0

    band_mask = (bpm_grid >= rr_min) & (bpm_grid <= rr_max)
    spec_filtered = np.zeros_like(spec)
    spec_filtered[band_mask] = spec[band_mask]

    # 反变换回时域，并只取前 n 个点
    sig_filt = np.fft.irfft(spec_filtered, n=n_fft)
    sig_filt = sig_filt[:n]

    # 把窗函数大致除回去（避免整体能量降低太多），防止极端震荡
    win_mean = np.mean(win ** 2)
    if win_mean > 1e-6:
        sig_filt = sig_filt / win_mean

    return sig_filt.astype(np.float64)


if __name__ == "__main__":
    # 简单自测：20 s 正弦波，看看估计出来的 RR 是否接近真值
    fs_demo = 30.0
    rr_true = 18.0    # bpm
    t = np.arange(0, 20, 1.0 / fs_demo)
    sig = np.sin(2 * np.pi * (rr_true / 60.0) * t)

    est = compute_rr_from_resp(sig, fs_demo, rr_min=6.0, rr_max=25.0, upsample_factor=4)
    print(f"[Demo] True RR={rr_true} bpm, Estimated RR={est:.2f} bpm")
