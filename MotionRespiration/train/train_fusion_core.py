# train/train_fusion_core.py
import os
import sys
import time
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    for b in range(B):
        p = pred[b]
        g = gt[b]
        best = torch.tensor(-1.0, device=pred.device)
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
    用简单 FFT 主峰估计批量 RR, 单位 BPM (训练内部用；eval 用 compute_rr_from_resp)
    signal: [B,T] 或 [B,1,T]
    """
    if signal.ndim == 3:
        signal = signal.squeeze(1)

    B, T = signal.shape
    device = signal.device

    sig = signal - signal.mean(dim=-1, keepdim=True)
    spec = torch.fft.rfft(sig, dim=-1)            # [B,F]
    mag = spec.abs()

    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(device)  # Hz
    bpm = freqs * 60.0

    band_mask = (bpm >= min_bpm) & (bpm <= max_bpm)
    if not band_mask.any():
        band_mask[:] = True

    mag_band = mag[:, band_mask]                 # [B, F_band]
    bpm_band = bpm[band_mask]                    # [F_band]

    idx = mag_band.argmax(dim=-1)                # [B]
    rr = bpm_band[idx]                           # [B]

    return rr


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
    denom = np.sqrt((vx ** 2).sum()) * np.sqrt((vy ** 2).sum()) + 1e-8
    if denom == 0:
        return 0.0
    c = float((vx * vy).sum() / denom)
    if np.isnan(c):
        return 0.0
    return c


# ==================== DataLoader ====================

def build_dataloaders(cfg):
    """
    使用 MotionDatasetFlow：
        返回:
            - clip_rgb: [3, T, H, W]
            - flow    : [2, T, H, W]
            - resp    : [T]
    """
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
        "corr_metric": 0.0,       # 硬对齐相关
        "corr_lag_metric": 0.0,   # 带时移最佳相关
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
            pred = model(rgb_clip, flow_clip)  # [B,T]
        else:
            pred = model(rgb_clip)            # [B,T]

        loss_dict = criterion_nmcc(pred, resp)
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()

        bsz = rgb_clip.size(0)
        n_samples += bsz

        with torch.no_grad():
            corr_each = pearson_corr_torch(pred, resp)               # 硬对齐
            corr_lag_each = corr_with_best_lag(pred, resp, 60)       # 带时移

            rr_pred_fft = estimate_rr_fft_batch(
                pred, cfg.FPS, cfg.MIN_BPM, cfg.MAX_BPM
            )
            rr_gt_fft = estimate_rr_fft_batch(
                resp, cfg.FPS, cfg.MIN_BPM, cfg.MAX_BPM
            )
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
                f"TrainTotal={running['total']/denom:.4f} "
                f"(NMCC={running['nmcc']/denom:.4f}, "
                f"AFD={running['afd']/denom:.4f}, "
                f"RR_L1={running['rr_l1']/denom:.4f}, "
                f"HighE={running['high_energy']/denom:.6f}), "
                f"CorrRaw={running['corr_metric']/denom:.4f}, "
                f"CorrLag={running['corr_lag_metric']/denom:.4f}, "
                f"RR_MAE_FFT={running['rr_mae_fft']/denom:.2f} bpm"
            )

    denom = max(1, n_samples)
    elapsed = time.time() - start
    for k in running.keys():
        running[k] /= denom

    print(
        f"[Fusion][Epoch {epoch}] Train done in {elapsed:.1f}s | "
        f"TrainTotal={running['total']:.4f} "
        f"(NMCC={running['nmcc']:.4f}, "
        f"AFD={running['afd']:.4f}, "
        f"RR_L1={running['rr_l1']:.4f}, "
        f"HighE={running['high_energy']:.6f}), "
        f"CorrRaw={running['corr_metric']:.4f}, "
        f"CorrLag={running['corr_lag_metric']:.4f}, "
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
    验证阶段：
    - 用 NMCCAFDLoss 计算 loss（不反传）
    - 同时计算两套 RR 指标：
        1) FFT 主峰版 (RR_FFT)
        2) compute_rr_from_resp 版 (RR_EVAL)
    - 打印两种相关:
        CorrRaw: 硬对齐 Pearson r
        CorrLag: 允许 ±60 帧时移后的最大相关
    """
    model.eval()
    device = cfg.DEVICE
    use_flow = getattr(cfg, "USE_FLOW", True)

    running = {
        "total": 0.0,
        "nmcc": 0.0,
        "afd": 0.0,
        "rr_l1": 0.0,
        "high_energy": 0.0,
        "corr_metric": 0.0,       # 硬对齐
        "corr_lag_metric": 0.0,   # 带时移
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
        loss = loss_dict["total"]

        bsz = rgb_clip.size(0)
        n_samples += bsz

        corr_each = pearson_corr_torch(pred, resp)
        corr_lag_each = corr_with_best_lag(pred, resp, 60)

        running["total"] += float(loss.item()) * bsz
        running["nmcc"] += float(loss_dict["nmcc"].item()) * bsz
        running["afd"] += float(loss_dict["afd"].item()) * bsz
        running["rr_l1"] += float(loss_dict["rr_l1"].item()) * bsz
        running["high_energy"] += float(loss_dict["high_energy"].item()) * bsz
        running["corr_metric"] += float(corr_each.mean().item()) * bsz
        running["corr_lag_metric"] += float(corr_lag_each.mean().item()) * bsz

        # ---------- 1) FFT 版 RR ----------
        rr_fft_pred = estimate_rr_fft_batch(
            pred, cfg.FPS, cfg.MIN_BPM, cfg.MAX_BPM
        )
        rr_fft_gt = estimate_rr_fft_batch(
            resp, cfg.FPS, cfg.MIN_BPM, cfg.MAX_BPM
        )
        all_rr_fft_pred.append(rr_fft_pred.cpu().numpy())
        all_rr_fft_gt.append(rr_fft_gt.cpu().numpy())

        # ---------- 2) compute_rr_from_resp 版 RR (和 eval_fusion 对齐) ----------
        pred_np = pred.detach().cpu().numpy()
        resp_np = resp.detach().cpu().numpy()
        for i in range(pred_np.shape[0]):
            rr_eval_gt = compute_rr_from_resp(
                resp_np[i], cfg.FPS, rr_min=cfg.MIN_BPM, rr_max=cfg.MAX_BPM
            )
            rr_eval_pred = compute_rr_from_resp(
                pred_np[i], cfg.FPS, rr_min=cfg.MIN_BPM, rr_max=cfg.MAX_BPM
            )
            all_rr_eval_gt.append(rr_eval_gt)
            all_rr_eval_pred.append(rr_eval_pred)

    # --------- 汇总 ----------
    denom = max(1, n_samples)
    elapsed = time.time() - start
    for k in running.keys():
        running[k] /= denom

    # FFT 版
    if len(all_rr_fft_gt) > 0:
        rr_fft_gt_all = np.concatenate(all_rr_fft_gt, axis=0)
        rr_fft_pred_all = np.concatenate(all_rr_fft_pred, axis=0)
        mae_fft = float(np.mean(np.abs(rr_fft_pred_all - rr_fft_gt_all)))
        rmse_fft = float(
            np.sqrt(np.mean((rr_fft_pred_all - rr_fft_gt_all) ** 2))
        )
        corr_fft = numpy_corr(rr_fft_pred_all, rr_fft_gt_all)
    else:
        mae_fft = rmse_fft = corr_fft = float("nan")

    # EVAL 版 (compute_rr_from_resp)
    if len(all_rr_eval_gt) > 0:
        rr_eval_gt_all = np.array(all_rr_eval_gt, dtype=np.float32)
        rr_eval_pred_all = np.array(all_rr_eval_pred, dtype=np.float32)
        mae_eval = float(np.mean(np.abs(rr_eval_pred_all - rr_eval_gt_all)))
        rmse_eval = float(
            np.sqrt(np.mean((rr_eval_pred_all - rr_eval_gt_all) ** 2))
        )
        corr_eval = numpy_corr(rr_eval_pred_all, rr_eval_gt_all)
    else:
        mae_eval = rmse_eval = corr_eval = float("nan")

    print(
        f"[Fusion][Epoch {epoch}] Val   | "
        f"ValTotal={running['total']:.4f}, "
        f"ValNMCC={running['nmcc']:.4f}, "
        f"ValAFD={running['afd']:.4f}, "
        f"ValRR_L1={running['rr_l1']:.4f}, "
        f"ValHighE={running['high_energy']:.6f}, "
        f"ValCorrRaw={running['corr_metric']:.4f}, "
        f"ValCorrLag={running['corr_lag_metric']:.4f}, "
        f"ValRR_FFT_MAE={mae_fft:.2f} bpm, "
        f"ValRR_FFT_RMSE={rmse_fft:.2f} bpm, "
        f"ValRR_FFT_Corr={corr_fft:.3f}, "
        f"ValRR_EVAL_MAE={mae_eval:.2f} bpm, "
        f"ValRR_EVAL_RMSE={rmse_eval:.2f} bpm, "
        f"ValRR_EVAL_Corr={corr_eval:.3f}, "
        f"time={elapsed:.1f}s"
    )

    # 返回两套 RR 指标, 方便外面选 best
    stats = dict(running)
    stats.update(
        rr_fft_mae=mae_fft,
        rr_fft_rmse=rmse_fft,
        rr_fft_corr=corr_fft,
        rr_eval_mae=mae_eval,
        rr_eval_rmse=rmse_eval,
        rr_eval_corr=corr_eval,
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

    print(
        "[Loss] NMCCAFDLoss cfg: "
        f"fps={cfg.FPS}, band=[{cfg.MIN_BPM},{cfg.MAX_BPM}] bpm, "
        f"alpha={cfg.NMCC_ALPHA}, afd_scale={cfg.AFD_SCALE}, "
        f"lambda_rr={cfg.LAMBDA_RR}, lambda_high={cfg.LAMBDA_HIGH}, "
        f"high_bpm={cfg.HIGH_BPM}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # 旧版：只按 best_mae_eval 选 best
    # 现在：用组合指标 score = mae_eval + lambda * max(0, target_corr - corr_eval)
    best_score = float("inf")
    best_epoch = -1

    print(
        f"[Fusion] Best model selection: score = mae_eval "
        f"+ {cfg.BEST_COMBO_LAMBDA} * max(0, {cfg.BEST_TARGET_CORR} - corr_eval)"
    )

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_stats = train_one_epoch(
            model, criterion_nmcc, optimizer, train_loader, epoch, cfg
        )
        val_stats = validate_one_epoch(
            model, criterion_nmcc, val_loader, epoch, cfg
        )

        mae_eval = val_stats["rr_eval_mae"]
        corr_eval = val_stats["rr_eval_corr"]

        # 组合评分：Corr 低于阈值时加罚，越小越好
        corr_gap = max(0.0, cfg.BEST_TARGET_CORR - corr_eval)
        score = mae_eval + cfg.BEST_COMBO_LAMBDA * corr_gap

        print(
            f"[Fusion] Epoch {epoch}: "
            f"TrainTotal={train_stats['total']:.4f}, "
            f"ValTotal={val_stats['total']:.4f}, "
            f"ValRR_FFT_MAE={val_stats['rr_fft_mae']:.2f}, "
            f"ValRR_EVAL_MAE={mae_eval:.2f}, "
            f"ValRR_EVAL_Corr={corr_eval:.3f}, "
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
                    "val_rr_eval_rmse": val_stats["rr_eval_rmse"],
                    "val_rr_eval_corr": corr_eval,
                    "val_rr_fft_mae": val_stats["rr_fft_mae"],
                    "val_rr_fft_rmse": val_stats["rr_fft_rmse"],
                    "val_rr_fft_corr": val_stats["rr_fft_corr"],
                    "val_corr_raw": val_stats["corr_metric"],
                    "val_corr_lag": val_stats["corr_lag_metric"],
                    # 把所有大写属性都存进去（包括 eval 的 step / 频带）
                    "cfg": {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
                },
                ckpt_path,
            )
            print(
                f"✅ [Fusion] New best model saved to: {ckpt_path} "
                f"(Score={best_score:.3f}, "
                f"ValRR_EVAL_MAE={mae_eval:.2f}, "
                f"ValRR_EVAL_Corr={corr_eval:.3f})"
            )

    print(
        f"[Fusion] Training finished. "
        f"Best epoch={best_epoch}, best Score={best_score:.3f}"
    )
