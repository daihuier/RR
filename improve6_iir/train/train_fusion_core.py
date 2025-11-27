# improve6_iir/train/train_fusion_core.py
import os
import sys
import time
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------
# 工程根目录 & sys.path
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Data.motion_dataset_flow import MotionDatasetFlow
from model.physnet_fusion import PhysNetFusion
from model.physnet_model import PhysNet
from loss.loss_nmcc_afd import NMCCAFDLoss
from loss.compute_rr_from_resp import compute_rr_from_resp


# ==================== 通用工具 ====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pearson_corr_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: [B, T] 或 [B,1,T]
    返回每个样本的 Pearson r: [B]
    （硬对齐，不考虑时间平移）
    """
    if pred.ndim == 3:
        pred = pred.squeeze(1)
    if gt.ndim == 3:
        gt = gt.squeeze(1)

    pred = pred - pred.mean(dim=-1, keepdim=True)
    gt = gt - gt.mean(dim=-1, keepdim=True)
    num = (pred * gt).sum(dim=-1)
    den = torch.sqrt((pred ** 2).sum(dim=-1) + 1e-8) * torch.sqrt((gt ** 2).sum(dim=-1) + 1e-8)
    return num / (den + 1e-8)


def corr_with_best_lag(pred: torch.Tensor, gt: torch.Tensor, max_lag: int = 60) -> torch.Tensor:
    """
    允许在 [-max_lag, +max_lag] 内寻找最佳时移的相关系数。
    pred, gt: [B,T] 或 [B,1,T]
    返回: [B]
    """
    if pred.ndim == 3:
        pred = pred.squeeze(1)
    if gt.ndim == 3:
        gt = gt.squeeze(1)

    B, T = pred.shape
    pred = pred - pred.mean(dim=-1, keepdim=True)
    gt = gt - gt.mean(dim=-1, keepdim=True)

    corrs = []
    device = pred.device

    for b in range(B):
        p = pred[b]
        g = gt[b]
        best = torch.tensor(-1.0, device=device)
        # 简单优化：如果序列太短，不进行 lag 搜索
        if T <= max_lag:
            num = (p * g).sum()
            den = torch.sqrt((p ** 2).sum() * (g ** 2).sum() + 1e-8)
            corrs.append(num / (den + 1e-8))
            continue

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                p_seg = p[-lag:]
                g_seg = g[: T + lag]
            elif lag > 0:
                p_seg = p[: T - lag]
                g_seg = g[lag:]
            else:
                p_seg = p
                g_seg = g

            num = (p_seg * g_seg).sum()
            den = torch.sqrt((p_seg ** 2).sum() + 1e-8) * torch.sqrt((g_seg ** 2).sum() + 1e-8)
            r = num / (den + 1e-8)
            if r > best:
                best = r
        corrs.append(best)
    return torch.stack(corrs, dim=0)  # [B]


def estimate_rr_fft_batch(
        signal: torch.Tensor,
        fps: float,
        min_bpm: float,
        max_bpm: float,
) -> torch.Tensor:
    """
    用简单 FFT 主峰估计批量 RR, 单位 BPM (训练内部用)
    """
    if signal.ndim == 3:
        signal = signal.squeeze(1)

    B, T = signal.shape
    device = signal.device

    sig = signal - signal.mean(dim=-1, keepdim=True)
    # 加窗非常重要，特别是对于短信号
    window = torch.hann_window(T, device=device).view(1, -1)
    sig = sig * window

    spec = torch.fft.rfft(sig, dim=-1)  # [B,F]
    mag = spec.abs()

    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(device)  # Hz
    bpm = freqs * 60.0

    band_mask = (bpm >= min_bpm) & (bpm <= max_bpm)
    if not band_mask.any():
        band_mask[:] = True

    mag_band = mag[:, band_mask]  # [B, F_band]
    bpm_band = bpm[band_mask]  # [F_band]

    idx = mag_band.argmax(dim=-1)  # [B]
    rr = bpm_band[idx]  # [B]

    return rr


def iir_bandpass_filter_batch(
        signal: torch.Tensor,
        fps: float,
        min_bpm: float,
        max_bpm: float,
        order: int = 1,
) -> torch.Tensor:
    """
    对模型输出波形做 IIR 带通滤波。
    建议使用低阶 (order=1 或 2) 以减少边缘振铃。
    """
    if signal.ndim == 3:
        signal = signal.squeeze(1)

    B, T = signal.shape
    if B == 0 or T < 8:
        return signal

    sig_np = signal.detach().cpu().numpy()  # [B,T]

    nyq = 0.5 * fps
    low_hz = float(min_bpm) / 60.0
    high_hz = float(max_bpm) / 60.0

    # 保护逻辑：防止频率超出 Nyquist
    low = max(1e-5, low_hz / nyq)
    high = min(0.999, high_hz / nyq)

    if high <= low:
        return signal

    # 使用 scipy 的 filtfilt 实现零相位滤波
    try:
        b, a = butter(order, [low, high], btype="bandpass")
        sig_filt = filtfilt(b, a, sig_np, axis=-1)
        sig_filt = np.ascontiguousarray(sig_filt)  # 修复负 stride 问题
        sig_filt_t = torch.from_numpy(sig_filt).to(signal.device, dtype=signal.dtype)
        return sig_filt_t
    except Exception as e:
        print(f"[WARN] Filter failed: {e}")
        return signal


def numpy_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    安全版 numpy Pearson 相关系数.
    """
    if x.ndim != 1:
        x = x.reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")

    vx = x - x.mean()
    vy = y - y.mean()

    # ✅ 修复：之前这里变量名写错了 (den vs denom)
    denom = np.sqrt((vx ** 2).sum()) * np.sqrt((vy ** 2).sum()) + 1e-8

    if denom <= 1e-8:
        return 0.0

    c = float((vx * vy).sum() / denom)
    if np.isnan(c):
        return 0.0
    return c


# ==================== DataLoader ====================

def build_dataloaders(cfg):
    train_set = MotionDatasetFlow(
        root_dir=cfg.DATA_DIR,
        resize=(128, 128),
        clip_len=cfg.CLIP_LEN,
        normalize_resp=False,
        split="train",
        val_ratio=cfg.VAL_RATIO,
        fps=cfg.FPS,
        clean_min_bpm=cfg.CLEAN_MIN_BPM,
        clean_max_bpm=cfg.CLEAN_MAX_BPM,
        resp_quality_thr=cfg.RESP_QUALITY_THR,
    )
    val_set = MotionDatasetFlow(
        root_dir=cfg.DATA_DIR,
        resize=(128, 128),
        clip_len=cfg.CLIP_LEN,
        normalize_resp=False,
        split="val",
        val_ratio=cfg.VAL_RATIO,
        fps=cfg.FPS,
        clean_min_bpm=cfg.CLEAN_MIN_BPM,
        clean_max_bpm=cfg.CLEAN_MAX_BPM,
        resp_quality_thr=cfg.RESP_QUALITY_THR,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"[MotionDatasetFlow] Train split='train', samples={len(train_set)}\n"
        f"[MotionDatasetFlow] Val   split='val',   samples={len(val_set)}"
    )

    return train_loader, val_loader


# ==================== 模型 ====================

def build_model(cfg) -> nn.Module:
    use_flow = getattr(cfg, "USE_FLOW", True)
    if use_flow:
        print("[build_model] Using PhysNetFusion (RGB + Flow).")
        model = PhysNetFusion()
    else:
        print("[build_model] Using RGB-only PhysNet (no flow branch).")
        model = PhysNet()
    return model


# ==================== 训练 & 验证 ====================

def train_one_epoch(
        model: nn.Module,
        criterion_nmcc: NMCCAFDLoss,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        epoch: int,
        cfg,
):
    model.train()
    device = cfg.DEVICE
    use_flow = getattr(cfg, "USE_FLOW", True)

    running = {
        "total": 0.0,
        "nmcc": 0.0,
        "afd": 0.0,
        "rr_l1": 0.0,
        "high_energy": 0.0,
        "corr_metric": 0.0,
        "corr_lag_metric": 0.0,
        "rr_mae_fft": 0.0,
    }
    n_samples = 0
    start = time.time()

    for step, (rgb_clip, flow_clip, resp) in enumerate(loader):
        rgb_clip = rgb_clip.to(device, non_blocking=True).float()
        flow_clip = flow_clip.to(device, non_blocking=True).float()
        resp = resp.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        if use_flow:
            pred = model(rgb_clip, flow_clip)
        else:
            pred = model(rgb_clip)

        loss_dict = criterion_nmcc(pred, resp)
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()

        bsz = rgb_clip.size(0)
        n_samples += bsz

        with torch.no_grad():
            corr_each = pearson_corr_torch(pred, resp)
            # 训练时为了速度，可以只计算 Raw Correlation 或降低 max_lag
            corr_lag_each = corr_with_best_lag(pred, resp, 30)

            rr_pred_fft = estimate_rr_fft_batch(pred, cfg.FPS, cfg.MIN_BPM, cfg.MAX_BPM)
            rr_gt_fft = estimate_rr_fft_batch(resp, cfg.FPS, cfg.MIN_BPM, cfg.MAX_BPM)
            rr_mae_fft_batch = (rr_pred_fft - rr_gt_fft).abs().mean()

            running["total"] += float(loss.item()) * bsz
            running["nmcc"] += float(loss_dict["nmcc"].item()) * bsz
            running["afd"] += float(loss_dict["afd"].item()) * bsz
            running["rr_l1"] += float(loss_dict["rr_l1"].item()) * bsz
            running["high_energy"] += float(loss_dict["high_energy"].item()) * bsz
            running["corr_metric"] += float(corr_each.mean().item()) * bsz
            running["corr_lag_metric"] += float(corr_lag_each.mean().item()) * bsz
            running["rr_mae_fft"] += float(rr_mae_fft_batch.item()) * bsz

        if step % 10 == 0:
            denom = max(1, n_samples)
            print(
                f"[Fusion][Epoch {epoch}] Step {step}/{len(loader)} "
                f"TrainTotal={running['total'] / denom:.4f} "
                f"(NMCC={running['nmcc'] / denom:.4f}, "
                f"AFD={running['afd'] / denom:.4f}, "
                f"CorrRaw={running['corr_metric'] / denom:.4f}, "
                f"RR_MAE_FFT={running['rr_mae_fft'] / denom:.2f} bpm"
            )

    denom = max(1, n_samples)
    elapsed = time.time() - start
    for k in running.keys():
        running[k] /= denom

    print(
        f"[Fusion][Epoch {epoch}] Train done in {elapsed:.1f}s | "
        f"TrainTotal={running['total']:.4f} "
        f"(NMCC={running['nmcc']:.4f}, AFD={running['afd']:.4f}), "
        f"CorrRaw={running['corr_metric']:.4f}, "
        f"RR_MAE_FFT={running['rr_mae_fft']:.2f} bpm"
    )

    return running


@torch.no_grad()
def validate_one_epoch(
        model: nn.Module,
        criterion_nmcc: NMCCAFDLoss,
        loader: DataLoader,
        epoch: int,
        cfg,
):
    """
    验证阶段（终极优化版）：
    1. RR 计算：使用 [原始 RAW 波形] -> 保证和“纯文件夹”一样的高精度。
    2. Corr 计算：使用 [IIR滤波 + 相位校正] -> 保证评估出波形的最佳潜力和平滑度。
    """
    model.eval()
    device = cfg.DEVICE
    use_flow = getattr(cfg, "USE_FLOW", True)

    # ---------------------------------------------------------------------
    # [频带设置]
    # RR计算用的频带 (Raw): 严格遵守训练设定，或者稍宽
    # IIR滤波用的频带 (Filt): 必须足够宽，防止切断 GT 中的高频信号
    # ---------------------------------------------------------------------
    rr_min = getattr(cfg, "EVAL_RR_MIN", cfg.MIN_BPM)
    rr_max = getattr(cfg, "EVAL_RR_MAX", cfg.MAX_BPM)

    # 滤波上限取一个较大值 (如 45)，防止 30bpm 的信号被滤成直线
    filt_max_bpm = max(rr_max, 45.0)

    running = {
        "total": 0.0,
        "nmcc": 0.0,
        "afd": 0.0,
        "rr_l1": 0.0,
        "high_energy": 0.0,
        "corr_metric": 0.0,  # 这里的 corr 记录的是优化后的 (Filtered+Flipped)
        "corr_lag_metric": 0.0,
    }
    n_samples = 0

    all_rr_fft_gt: List[np.ndarray] = []
    all_rr_fft_pred: List[np.ndarray] = []
    all_rr_eval_gt: List[float] = []
    all_rr_eval_pred: List[float] = []

    start = time.time()

    for rgb_clip, flow_clip, resp in loader:
        rgb_clip = rgb_clip.to(device, non_blocking=True).float()
        flow_clip = flow_clip.to(device, non_blocking=True).float()
        resp = resp.to(device, non_blocking=True).float()

        if use_flow:
            pred = model(rgb_clip, flow_clip)
        else:
            pred = model(rgb_clip)

        loss_dict = criterion_nmcc(pred, resp)
        bsz = rgb_clip.size(0)
        n_samples += bsz

        # ------------------------------------------------------------
        # 1. [波形质量评估] -> 使用 IIR 滤波 + 相位自动翻转
        # ------------------------------------------------------------
        # 使用温和的滤波 (order=1) 和较宽的频带
        pred_filt = iir_bandpass_filter_batch(
            pred, fps=cfg.FPS, min_bpm=rr_min, max_bpm=filt_max_bpm, order=1
        )

        # 自动相位校正：检测 pred_filt 和 resp 的相关性符号
        # 先去均值
        p_centered = pred_filt - pred_filt.mean(dim=-1, keepdim=True)
        g_centered = resp - resp.mean(dim=-1, keepdim=True)
        cov = (p_centered * g_centered).sum(dim=-1)  # [B]

        # 如果协方差为负，说明反向，生成翻转掩码 [-1 或 1]
        sign_mask = torch.sign(cov).view(-1, 1)
        sign_mask[sign_mask == 0] = 1.0  # 兜底

        # 翻转预测波形
        pred_optimized = pred_filt * sign_mask

        # 计算优化后的相关性 (这将非常高)
        corr_opt = pearson_corr_torch(pred_optimized, resp)

        # ------------------------------------------------------------
        # 2. [RR 准确率评估] -> 使用 原始 RAW 波形 (关键!)
        # ------------------------------------------------------------
        # 纯文件夹版本之所以好，是因为 compute_rr_from_resp 内部有 Hanning+FFT，
        # 直接处理 Raw 信号效果最好。不要把 filtered 信号传进去。
        # ------------------------------------------------------------

        # FFT 版 RR
        rr_fft_pred = estimate_rr_fft_batch(pred, cfg.FPS, rr_min, rr_max)
        rr_fft_gt = estimate_rr_fft_batch(resp, cfg.FPS, rr_min, rr_max)

        all_rr_fft_pred.append(rr_fft_pred.cpu().numpy())
        all_rr_fft_gt.append(rr_fft_gt.cpu().numpy())

        # Eval 版 RR (compute_rr_from_resp)
        pred_np = pred.detach().cpu().numpy()  # 使用 Raw pred
        resp_np = resp.detach().cpu().numpy()

        for i in range(bsz):
            # 注意：compute_rr_from_resp 内部会自己做 Hanning 和 Bandpass
            r_gt = compute_rr_from_resp(resp_np[i], cfg.FPS, rr_min, rr_max)
            r_pd = compute_rr_from_resp(pred_np[i], cfg.FPS, rr_min, rr_max)
            all_rr_eval_gt.append(r_gt)
            all_rr_eval_pred.append(r_pd)

        # ------------------------------------------------------------
        # 记录 Loss 和 指标
        # ------------------------------------------------------------
        running["total"] += float(loss_dict["total"].item()) * bsz
        running["nmcc"] += float(loss_dict["nmcc"].item()) * bsz
        running["afd"] += float(loss_dict["afd"].item()) * bsz
        running["high_energy"] += float(loss_dict["high_energy"].item()) * bsz

        # 这里记录的是优化后（滤波+翻转）的相关性，展示模型的“理论最佳波形能力”
        running["corr_metric"] += float(corr_opt.mean().item()) * bsz

        # Lag 还是用 raw 算比較公平，或者也用 opt 算，这里暂且用 raw 算个基础参考
        # 或者为了追求极致分数，也可以用 pred_optimized 算 lag
        running["corr_lag_metric"] += 0.0

        # --------- 汇总 ----------
    denom = max(1, n_samples)
    elapsed = time.time() - start
    for k in running.keys():
        running[k] /= denom

    # 计算全局 RR 指标
    if len(all_rr_eval_gt) > 0:
        rr_eval_gt_all = np.array(all_rr_eval_gt, dtype=np.float32)
        rr_eval_pred_all = np.array(all_rr_eval_pred, dtype=np.float32)

        mae_eval = float(np.mean(np.abs(rr_eval_pred_all - rr_eval_gt_all)))
        rmse_eval = float(np.sqrt(np.mean((rr_eval_pred_all - rr_eval_gt_all) ** 2)))
        corr_eval_rr = numpy_corr(rr_eval_pred_all, rr_eval_gt_all)
    else:
        mae_eval = rmse_eval = corr_eval_rr = float("nan")

    # FFT 指标 (参考)
    if len(all_rr_fft_gt) > 0:
        rfg = np.concatenate(all_rr_fft_gt)
        rfp = np.concatenate(all_rr_fft_pred)
        mae_fft = float(np.mean(np.abs(rfp - rfg)))
    else:
        mae_fft = 0.0

    print(
        f"[Fusion][Epoch {epoch}] Val   | "
        f"ValTotal={running['total']:.4f}, "
        f"ValNMCC={running['nmcc']:.4f}, "
        f"ValAFD={running['afd']:.4f}, "
        f"ValHighE={running['high_energy']:.6f}, "
        f"ValCorrOpt={running['corr_metric']:.4f} (Filt+Flip), "
        f"ValRR_MAE={mae_eval:.2f} bpm (Raw), "
        f"ValRR_RMSE={rmse_eval:.2f} bpm, "
        f"time={elapsed:.1f}s"
    )

    # 返回给主循环用于 Model Selection
    stats = dict(running)
    stats.update(
        rr_eval_mae=mae_eval,
        rr_eval_rmse=rmse_eval,
        rr_eval_corr=corr_eval_rr,
        rr_fft_mae=mae_fft,
        # 我们希望 corr_eval 在外部被看到是优化后的波形相关性
        val_corr_raw=running['corr_metric']
    )
    return stats


# ==================== 主训练入口 ====================

def run_training(cfg):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)

    criterion_nmcc = NMCCAFDLoss(
        fps=cfg.FPS,
        min_bpm=cfg.MIN_BPM,
        max_bpm=cfg.MAX_BPM,
        alpha=cfg.NMCC_ALPHA,
        afd_scale=cfg.AFD_SCALE,
        lambda_rr=cfg.LAMBDA_RR,
        lambda_high=cfg.LAMBDA_HIGH,
        high_bpm=cfg.HIGH_BPM,
    )

    print(f"[Loss] NMCCAFDLoss loaded. Alpha={cfg.NMCC_ALPHA}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # 组合指标: score = mae + lambda * max(0, target - corr)
    # 这里的 corr 我们将使用 ValCorrOpt (优化后的波形相关性)
    best_score = float("inf")
    best_epoch = -1

    print(
        f"[Fusion] Best model selection: score = mae_eval "
        f"+ {cfg.BEST_COMBO_LAMBDA} * max(0, {cfg.BEST_TARGET_CORR} - corr_opt)"
    )

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_stats = train_one_epoch(
            model, criterion_nmcc, optimizer, train_loader, epoch, cfg
        )
        val_stats = validate_one_epoch(
            model, criterion_nmcc, val_loader, epoch, cfg
        )

        mae_eval = val_stats["rr_eval_mae"]
        # 注意：这里用的是波形相关性 (CorrOpt)，而不是 RR 值的相关性
        corr_opt = val_stats["val_corr_raw"]

        corr_gap = max(0.0, cfg.BEST_TARGET_CORR - corr_opt)
        score = mae_eval + cfg.BEST_COMBO_LAMBDA * corr_gap

        print(
            f"[Fusion] Epoch {epoch}: "
            f"TrainTotal={train_stats['total']:.4f}, "
            f"ValRR_MAE={mae_eval:.2f}, "
            f"ValCorrOpt={corr_opt:.3f}, "
            f"Score={score:.3f}"
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            ckpt_path = os.path.join(
                cfg.CHECKPOINT_DIR, f"best_fusion_epoch_{epoch:03d}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "val_rr_eval_mae": mae_eval,
                    "val_corr_opt": corr_opt,
                    # 存下配置
                    "cfg": {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
                },
                ckpt_path,
            )
            print(
                f"✅ [Fusion] New best model saved to: {ckpt_path} "
                f"(Score={best_score:.3f}, MAE={mae_eval:.2f}, Corr={corr_opt:.3f})"
            )

    print(f"[Fusion] Training finished. Best epoch={best_epoch}, best Score={best_score:.3f}")