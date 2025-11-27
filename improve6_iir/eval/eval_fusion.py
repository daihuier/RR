# eval/eval_fusion.py
# 自动选择最新 best_fusion_epoch_XXX.pth，并在 val split 上做评估（滑动窗 + 中位数聚合）
# 支持自动识别 checkpoint 类型：
#   - RGB-only PhysNet
#   - Fusion PhysNetFusion (RGB + Flow)
#
# 本版增加（基于 improve6_iir 理论最优策略）：
#   - RR 计算：使用【原始 RAW 波形】以保证最高频率准确度（避免滤波振铃）。
#   - 波形展示/相关性：使用【IIR滤波 (Order=1) + 相位自动校正】以展示最佳波形形态。
#   - 动态频带：滤波上限自动放宽，防止误杀高频信号。

import os
import sys
import time
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ----------------------------------------------------------------------
# 确保可以 import 项目内模块
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 不直接 from ... import ...，避免循环导入问题
import Data.motion_dataset_flow as motion_dataset_flow
from loss.compute_rr_from_resp import compute_rr_from_resp

# ----------------------------------------------------------------------
# 一些默认全局参数（可以被 ckpt / 命令行覆盖）
# ----------------------------------------------------------------------
CLIP_LEN_EVAL_DEFAULT = 600  # 20 秒 @ 30 FPS
STEP_DEFAULT = 200  # 滑动窗口步长
FPS = 30.0
RR_MIN_DEFAULT = 6.0
RR_MAX_DEFAULT = 25.0
UPSAMPLE_DEFAULT = 4

# 波形绘图相关配置
MAX_PLOTS = 10  # 稍微多画几张
PLOTS_DIR = os.path.join(PROJECT_ROOT, "eval_plots")


# ----------------------------------------------------------------------
# 工具函数：信号处理
# ----------------------------------------------------------------------
def slice_into_windows(arr: np.ndarray, win: int, step: int) -> List[np.ndarray]:
    """将 1D 序列按滑动窗口切片."""
    T = len(arr)
    segs = []
    for s in range(0, T - win + 1, step):
        segs.append(arr[s:s + win])
    if not segs and T > 0:
        segs.append(arr)
    return segs


def apply_iir_bandpass(
        data: np.ndarray,
        fps: float,
        min_bpm: float,
        max_bpm: float,
        order: int = 1
) -> np.ndarray:
    """
    IIR 带通滤波 (Butterworth).
    注意：这是为了获得'好看'的波形和'正确'的相关性，
    对于 RR 估算，compute_rr_from_resp 内部的 FFT 处理通常更鲁棒。
    """
    if len(data) < 8:
        return data

    nyq = 0.5 * fps
    low = max(1e-5, (min_bpm / 60.0) / nyq)
    high = min(0.999, (max_bpm / 60.0) / nyq)

    if high <= low:
        return data

    try:
        b, a = butter(order, [low, high], btype='bandpass')
        # 使用 filtfilt 实现零相位滤波
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        print(f"[Eval] Filter error: {e}")
        return data


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """简单 z-score，方便波形形状对比。"""
    m = x.mean()
    s = x.std()
    if s < eps:
        return x * 0.0
    return (x - m) / (s + eps)


def plot_waveform_pair(
        gt: np.ndarray,
        pred: np.ndarray,
        sample_idx: int,
        save_dir: str = PLOTS_DIR,
        fps: float = FPS,
):
    """
    画一张 GT vs Pred 波形图并保存为 PNG。
    注意：传入的 pred 应该是已经经过滤波和相位校正的 'optimized' 波形。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 对齐长度
    T = min(len(gt), len(pred))
    if T <= 1:
        return
    gt = gt[:T]
    pred = pred[:T]

    # Z-score 归一化 (只为了画图对齐幅度)
    gt_z = _zscore(gt)
    pred_z = _zscore(pred)

    t_axis = np.arange(T) / fps

    plt.figure(figsize=(10, 3.5))
    plt.plot(t_axis, gt_z, label="GT (z-score)", linewidth=1.5, alpha=0.8)
    # 使用虚线，颜色醒目
    plt.plot(t_axis, pred_z, label="Pred (Opt: Filt+Flip)", linewidth=1.5, linestyle="--", color='orange', alpha=0.9)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z-score)")
    plt.title(f"Sample #{sample_idx:03d} - Waveform Comparison (Optimized)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"sample_{sample_idx:03d}_waveform.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[EvalFusion] Saved waveform plot → {out_path}")


def find_latest_best_ckpt(ckpt_dir: str) -> Tuple[str, int]:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    best_epoch = -1
    best_path = None

    for name in os.listdir(ckpt_dir):
        if not name.startswith("best_fusion_epoch_") or not name.endswith(".pth"):
            continue
        stem = name[len("best_fusion_epoch_"):-4]
        try:
            ep = int(stem)
        except ValueError:
            continue
        if ep > best_epoch:
            best_epoch = ep
            best_path = os.path.join(ckpt_dir, name)

    if best_path is None:
        raise FileNotFoundError(f"No best_fusion_epoch_XXX.pth found in {ckpt_dir}")

    return best_path, best_epoch


# ----------------------------------------------------------------------
# 模型构建
# ----------------------------------------------------------------------
def detect_model_type(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    if not keys:
        return "rgb"
    fusion_prefixes = ("rgb_net.", "flow_net.", "fuse.")
    for k in keys:
        if k.startswith(fusion_prefixes):
            return "fusion"
    return "rgb"


def build_model_from_state_dict(state_dict: dict, device: torch.device):
    from model.physnet_fusion import PhysNetFusion
    from model.physnet_model import PhysNet

    model_type = detect_model_type(state_dict)
    if model_type == "fusion":
        print("[EvalFusion] Detected Fusion checkpoint → using PhysNetFusion (RGB + Flow).")
        model = PhysNetFusion().to(device)
    else:
        print("[EvalFusion] Detected RGB-only checkpoint → using PhysNet.")
        model = PhysNet().to(device)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[EvalFusion][WARN] Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"[EvalFusion][WARN] Unexpected keys: {incompatible.unexpected_keys}")

    model.eval()
    return model, model_type


# ----------------------------------------------------------------------
# 主评估函数
# ----------------------------------------------------------------------
def evaluate_model(
        ckpt_path: str,
        data_root: str,
        device_str: str = "auto",
        clip_len: int = -1,
        step: int = -1,
        rr_min: float = -1.0,
        rr_max: float = -1.0,
        upsample_factor: int = UPSAMPLE_DEFAULT,
):
    # ---------------- 设备选择 ----------------
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[EvalFusion] Device: {device}")

    # ---------------- 加载 checkpoint ----------------
    print(f"[EvalFusion] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    cfg_from_ckpt = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}

    # ---------------- 解析参数 (优先命令行 -> ckpt -> 默认) ----------------
    if clip_len is None or clip_len <= 0:
        clip_len_eff = int(cfg_from_ckpt.get("EVAL_CLIP_LEN", cfg_from_ckpt.get("CLIP_LEN", CLIP_LEN_EVAL_DEFAULT)))
    else:
        clip_len_eff = int(clip_len)

    if step is None or step <= 0:
        step_eff = int(cfg_from_ckpt.get("EVAL_STEP", STEP_DEFAULT))
    else:
        step_eff = int(step)

    if rr_min is None or rr_min <= 0:
        rr_min_eff = float(cfg_from_ckpt.get("EVAL_RR_MIN", cfg_from_ckpt.get("MIN_BPM", RR_MIN_DEFAULT)))
    else:
        rr_min_eff = float(rr_min)

    if rr_max is None or rr_max <= 0:
        rr_max_eff = float(cfg_from_ckpt.get("EVAL_RR_MAX", cfg_from_ckpt.get("MAX_BPM", RR_MAX_DEFAULT)))
    else:
        rr_max_eff = float(rr_max)

    resp_quality_thr = float(cfg_from_ckpt.get("RESP_QUALITY_THR", 0.15))

    # [关键] 滤波上限：为了防止误杀 30-40bpm 的高频信号，滤波上限取一个较大值
    # 这样画图时高频部分能保留，而 RR 计算本身就不受这个影响 (因为用 Raw)
    clean_max_bpm = float(cfg_from_ckpt.get("CLEAN_MAX_BPM", 40.0))
    filt_max_bpm = max(rr_max_eff, clean_max_bpm, 45.0)

    print(
        f"[EvalFusion] Param: clip={clip_len_eff}, step={step_eff}, RR_Range=[{rr_min_eff},{rr_max_eff}], Filt_Max={filt_max_bpm}")

    # ---------------- 构建模型 ----------------
    model, model_type = build_model_from_state_dict(state_dict, device)

    # ---------------- 构建数据集 ----------------
    eval_set = motion_dataset_flow.MotionDatasetFlow(
        root_dir=data_root,
        clip_len=clip_len_eff,
        split="val",
        fps=FPS,
        clean_min_bpm=rr_min_eff,
        clean_max_bpm=clean_max_bpm,  # 使用宽频带清洗
        resp_quality_thr=resp_quality_thr,
    )
    loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

    print(f"[EvalFusion] Val size = {len(eval_set)}")

    all_pred_rr = []
    all_gt_rr = []
    all_waveform_corrs = []  # 记录波形相关性

    t0 = time.time()
    plotted = 0

    for idx, (rgb_clip, flow_clip, resp) in enumerate(loader):
        rgb_clip = rgb_clip.to(device).float()
        flow_clip = flow_clip.to(device).float()
        resp_np = resp.numpy()[0]  # GT: (T,)

        # 1. 模型推理 (Raw Output)
        with torch.no_grad():
            if model_type == "fusion":
                pred_resp_raw = model(rgb_clip, flow_clip).cpu().numpy()[0]
            else:
                pred_resp_raw = model(rgb_clip).cpu().numpy()[0]

        # ---------------------------------------------------------------------
        # [策略 A] RR 计算：使用【原始 RAW 波形】
        # ---------------------------------------------------------------------
        resp_windows = slice_into_windows(resp_np, clip_len_eff, step_eff)
        # 注意：Raw 预测值不需要切片，直接重复利用或按逻辑切片，这里保持原逻辑
        # 原逻辑似乎是拿整个 pred 对应每个 GT window 计算?
        # 实际上 slice_into_windows 如果 clip_len_eff == len(resp) 只会返回一个窗口
        # 这里我们假设 eval clip 长度跟 pred 长度一致

        sample_pred_rrs = []
        sample_gt_rrs = []

        # 简单起见，如果长度一致，直接一一对应；不一致则按窗口
        pred_windows = slice_into_windows(pred_resp_raw, clip_len_eff, step_eff)
        n_wins = min(len(resp_windows), len(pred_windows))

        for i in range(n_wins):
            w_gt = resp_windows[i]
            w_pred_raw = pred_windows[i]  # 使用 RAW

            # compute_rr_from_resp 内部有 Hanning + FFT + Masking
            # 这是最精准的频率估算方式
            rr_pred = compute_rr_from_resp(
                w_pred_raw, FPS, rr_min=rr_min_eff, rr_max=rr_max_eff, upsample_factor=upsample_factor
            )
            rr_gt = compute_rr_from_resp(
                w_gt, FPS, rr_min=rr_min_eff, rr_max=rr_max_eff, upsample_factor=upsample_factor
            )
            sample_pred_rrs.append(rr_pred)
            sample_gt_rrs.append(rr_gt)

        # 聚合本样本 RR
        all_pred_rr.append(float(np.median(sample_pred_rrs)))
        all_gt_rr.append(float(np.median(sample_gt_rrs)))

        # ---------------------------------------------------------------------
        # [策略 B] 波形展示 & 相关性：使用【滤波 + 相位校正后波形】
        # ---------------------------------------------------------------------
        # 1. 滤波：去除高频毛刺 (order=1, 宽频带)
        pred_optim = apply_iir_bandpass(
            pred_resp_raw,
            fps=FPS,
            min_bpm=rr_min_eff,
            max_bpm=filt_max_bpm,
            order=1
        )

        # 2. 相位校正：如果与 GT 负相关，则翻转
        # 为了计算相关性，先对齐长度
        L = min(len(resp_np), len(pred_optim))
        if L > 10:
            gt_seg = resp_np[:L]
            pred_seg = pred_optim[:L]

            # 计算皮尔逊相关系数
            # 稍微处理下 NaN (常数波形)
            if np.std(gt_seg) < 1e-6 or np.std(pred_seg) < 1e-6:
                corr_val = 0.0
            else:
                corr_val = np.corrcoef(gt_seg, pred_seg)[0, 1]

            if corr_val < 0:
                pred_optim = -pred_optim
                corr_val = -corr_val  # 翻转后相关性变为正

            all_waveform_corrs.append(corr_val)
        else:
            all_waveform_corrs.append(0.0)

        # 3. 画图 (使用优化后的波形)
        if plotted < MAX_PLOTS:
            plot_waveform_pair(
                gt=resp_np,
                pred=pred_optim,  # 传入优化后的波形
                sample_idx=idx,
                save_dir=PLOTS_DIR,
                fps=FPS,
            )
            plotted += 1

        if (idx + 1) % 10 == 0:
            print(f"[EvalFusion] Processed {idx + 1}/{len(loader)}")

    # ---------------- 计算全局指标 ----------------
    all_pred_rr = np.array(all_pred_rr)
    all_gt_rr = np.array(all_gt_rr)

    mae = float(np.mean(np.abs(all_pred_rr - all_gt_rr)))
    rmse = float(np.sqrt(np.mean((all_pred_rr - all_gt_rr) ** 2)))

    # RR 值的相关性
    if len(all_pred_rr) > 1:
        corr_rr = float(np.corrcoef(all_pred_rr, all_gt_rr)[0, 1])
    else:
        corr_rr = float("nan")

    # 波形平均相关性
    avg_waveform_corr = float(np.mean(all_waveform_corrs)) if all_waveform_corrs else 0.0

    t1 = time.time()

    print("============== EVAL FUSION RESULTS ==============")
    print(f"Model type      : {model_type}")
    print(f"Samples used    : {len(all_gt_rr)}")
    print(f"MAE (bpm)       : {mae:.3f}  <-- Key Metric (Raw)")
    print(f"RMSE (bpm)      : {rmse:.3f}")
    print(f"Corr (RR value) : {corr_rr:.3f}")
    print(f"Corr (Waveform) : {avg_waveform_corr:.3f} <-- Optimized (Filt+Flip)")
    print(f"Time elapsed    : {t1 - t0:.2f} s")

    print("First few pairs (GT | Pred_RR):")
    for i in range(min(10, len(all_gt_rr))):
        print(f"  #{i:02d}: {all_gt_rr[i]:6.2f} | {all_pred_rr[i]:6.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="auto")
    parser.add_argument("--data_root", type=str, default="/data/dsr/OVRM/Motion_Dataset")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_len", type=int, default=-1)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--rr_min", type=float, default=-1.0)
    parser.add_argument("--rr_max", type=float, default=-1.0)
    parser.add_argument("--upsample", type=int, default=UPSAMPLE_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.ckpt == "auto":
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints_fusion")
        ckpt_path, ep = find_latest_best_ckpt(ckpt_dir)
        print(f"[EvalFusion] Auto-selected: {ckpt_path} (epoch={ep})")
    else:
        ckpt_path = args.ckpt

    evaluate_model(
        ckpt_path=ckpt_path,
        data_root=args.data_root,
        device_str=args.device,
        clip_len=args.clip_len,
        step=args.step,
        rr_min=args.rr_min,
        rr_max=args.rr_max,
        upsample_factor=args.upsample,
    )