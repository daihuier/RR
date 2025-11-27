# eval/eval_fusion.py
# 自动选择最新 best_fusion_epoch_XXX.pth，并在 val split 上做评估（滑动窗 + 中位数聚合）
# 支持自动识别 checkpoint 类型：
#   - RGB-only PhysNet
#   - Fusion PhysNetFusion (RGB + Flow)
#
# 本版增加：
#   - clip_len / step / RR 频带 / upsample_factor 可配置
#   - 默认优先读 ckpt["cfg"] 里的配置，保持和训练一致
#   - 新增：前若干个样本的 GT vs Pred 波形图保存到 eval_plots/

import os
import sys
import time
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # ✅ 用于画波形图

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
CLIP_LEN_EVAL_DEFAULT = 600   # 20 秒 @ 30 FPS
STEP_DEFAULT = 200            # 滑动窗口步长
FPS = 30.0
RR_MIN_DEFAULT = 6.0
RR_MAX_DEFAULT = 25.0
UPSAMPLE_DEFAULT = 4

# ✅ 波形绘图相关配置
MAX_PLOTS = 5  # 最多画前多少个样本的波形对比
PLOTS_DIR = os.path.join(PROJECT_ROOT, "eval_plots")


def slice_into_windows(arr: np.ndarray, win: int, step: int) -> List[np.ndarray]:
    """将 1D 序列按滑动窗口切片."""
    T = len(arr)
    segs = []
    for s in range(0, T - win + 1, step):
        segs.append(arr[s:s + win])
    # 如果太短完全切不出窗口，可以直接返回整个序列作为一段
    if not segs and T > 0:
        segs.append(arr)
    return segs


def find_latest_best_ckpt(ckpt_dir: str) -> Tuple[str, int]:
    """
    在 ckpt_dir 里自动找到编号最大的 best_fusion_epoch_XXX.pth
    返回 (path, epoch_id)，如果找不到则报错。
    """
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
# 自动识别 checkpoint 类型：RGB-only 或 Fusion
# ----------------------------------------------------------------------
def detect_model_type(state_dict: dict) -> str:
    """
    根据 state_dict 的 key 自动判断模型类型：
      - 返回 "fusion" 或 "rgb"
    规则：
      - 如果有 'rgb_net.' 或 'flow_net.' 或 'fuse.' 前缀 → Fusion
      - 否则默认认为是 RGB-only PhysNet
    """
    keys = list(state_dict.keys())
    if not keys:
        return "rgb"

    # 只要有任意 key 带这些前缀，就认为是 fusion
    fusion_prefixes = ("rgb_net.", "flow_net.", "fuse.")
    for k in keys:
        if k.startswith(fusion_prefixes):
            return "fusion"

    # 否则默认 RGB-only
    return "rgb"


def build_model_from_state_dict(state_dict: dict, device: torch.device):
    """
    根据 state_dict 的 key 自动选择模型结构并加载权重。
    返回:
        model: 已经 load_state_dict 并 .eval() 的模型
        model_type: "fusion" 或 "rgb"
    """
    from model.physnet_fusion import PhysNetFusion
    from model.physnet_model import PhysNet

    model_type = detect_model_type(state_dict)
    if model_type == "fusion":
        print("[EvalFusion] Detected Fusion checkpoint → using PhysNetFusion (RGB + Flow).")
        model = PhysNetFusion().to(device)
    else:
        print("[EvalFusion] Detected RGB-only checkpoint → using PhysNet.")
        model = PhysNet().to(device)

    # 用 strict=False，避免老版本/新版本 key 不完全匹配时直接崩
    incompatible = model.load_state_dict(state_dict, strict=False)

    # 友好打印一下缺失/多余的 key，方便你调试
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])

    if missing:
        print("[EvalFusion][WARNING] Missing keys when loading state_dict:")
        for k in missing:
            print("   (missing)", k)
    if unexpected:
        print("[EvalFusion][WARNING] Unexpected keys in state_dict (ignored):")
        for k in unexpected:
            print("   (unexpected)", k)

    model.eval()
    print(f"[EvalFusion] Final model device: {next(model.parameters()).device}")
    print(f"[EvalFusion] Model type: {model_type}")
    return model, model_type


# ----------------------------------------------------------------------
# ✅ 新增：波形绘制辅助函数
# ----------------------------------------------------------------------
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
    - gt, pred: 1D numpy array
    - sample_idx: 样本编号，用于文件名
    """
    os.makedirs(save_dir, exist_ok=True)

    # 对齐长度：取两者最短长度
    T = min(len(gt), len(pred))
    if T <= 1:
        return
    gt = gt[:T]
    pred = pred[:T]

    # 做 z-score，让形状对比更清楚
    gt_z = _zscore(gt)
    pred_z = _zscore(pred)

    t_axis = np.arange(T) / fps  # 以秒为横坐标

    plt.figure(figsize=(8, 3))
    plt.plot(t_axis, gt_z, label="GT (z-score)", linewidth=1.0)
    plt.plot(t_axis, pred_z, label="Pred (z-score)", linewidth=1.0, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z-score)")
    plt.title(f"Sample #{sample_idx:03d} - Waveform Comparison")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"sample_{sample_idx:03d}_waveform.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[EvalFusion] Saved waveform plot → {out_path}")


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
    print(f"[EvalFusion] Initial device request: {device_str} -> Using {device}")

    # ---------------- 加载 checkpoint ----------------
    print(f"[EvalFusion] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容只存 model_state_dict 或带 'model_state_dict' 键两种情况
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # 从 ckpt 中取训练时的 cfg（如果有）
    cfg_from_ckpt = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}

    # ---------------- 解析 eval 超参（优先命令行，其次 ckpt.cfg，最后默认） ----------------
    if clip_len is None or clip_len <= 0:
        clip_len_eff = int(
            cfg_from_ckpt.get(
                "EVAL_CLIP_LEN",
                cfg_from_ckpt.get("CLIP_LEN", CLIP_LEN_EVAL_DEFAULT),
            )
        )
    else:
        clip_len_eff = int(clip_len)

    if step is None or step <= 0:
        step_eff = int(cfg_from_ckpt.get("EVAL_STEP", STEP_DEFAULT))
    else:
        step_eff = int(step)

    if rr_min is None or rr_min <= 0:
        rr_min_eff = float(
            cfg_from_ckpt.get("EVAL_RR_MIN", cfg_from_ckpt.get("MIN_BPM", RR_MIN_DEFAULT))
        )
    else:
        rr_min_eff = float(rr_min)

    if rr_max is None or rr_max <= 0:
        rr_max_eff = float(
            cfg_from_ckpt.get("EVAL_RR_MAX", cfg_from_ckpt.get("MAX_BPM", RR_MAX_DEFAULT))
        )
    else:
        rr_max_eff = float(rr_max)

    resp_quality_thr = float(cfg_from_ckpt.get("RESP_QUALITY_THR", 0.15))

    print(
        f"[EvalFusion] clip_len={clip_len_eff}, step={step_eff}, "
        f"RR band=[{rr_min_eff},{rr_max_eff}] bpm, "
        f"upsample_factor={upsample_factor}, "
        f"resp_quality_thr={resp_quality_thr}"
    )

    # ---------------- 构建模型（自动识别 RGB-only / Fusion）----------------
    model, model_type = build_model_from_state_dict(state_dict, device)

    # ---------------- 构建数据集（val split）----------------
    eval_set = motion_dataset_flow.MotionDatasetFlow(
        root_dir=data_root,
        clip_len=clip_len_eff,
        split="val",
        fps=FPS,
        clean_min_bpm=rr_min_eff,
        clean_max_bpm=rr_max_eff,
        resp_quality_thr=resp_quality_thr,
    )
    loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

    print(f"[EvalFusion] MotionDatasetFlow val size = {len(eval_set)}")

    all_pred_rr = []
    all_gt_rr = []

    t0 = time.time()

    # ✅ 计数：只对前 MAX_PLOTS 个样本画波形
    plotted = 0

    for idx, (rgb_clip, flow_clip, resp) in enumerate(loader):
        rgb_clip = rgb_clip.to(device).float()
        flow_clip = flow_clip.to(device).float()
        resp_np = resp.numpy()[0]  # (T,)

        # resp 做滑窗，对齐 clip_len_eff 窗口
        resp_windows = slice_into_windows(resp_np, clip_len_eff, step_eff)

        pred_rr_windows = []
        gt_rr_windows = []

        with torch.no_grad():
            if model_type == "fusion":
                # Fusion 模型：需要 RGB + Flow
                pred_resp_np = model(rgb_clip, flow_clip).cpu().numpy()[0]  # (T',)
            else:
                # RGB-only 模型：只用 RGB
                pred_resp_np = model(rgb_clip).cpu().numpy()[0]  # (T',)

        # ✅ 对前几个样本画波形：GT vs Pred（用完整 resp_np & pred_resp_np）
        if plotted < MAX_PLOTS:
            plot_waveform_pair(
                gt=resp_np,
                pred=pred_resp_np,
                sample_idx=idx,
                save_dir=PLOTS_DIR,
                fps=FPS,
            )
            plotted += 1

        for w in resp_windows:
            # 简化处理：同一个 pred_resp 对多个 GT 窗口
            rr_pred = compute_rr_from_resp(
                pred_resp_np,
                FPS,
                rr_min=rr_min_eff,
                rr_max=rr_max_eff,
                upsample_factor=upsample_factor,
            )
            rr_gt = compute_rr_from_resp(
                w,
                FPS,
                rr_min=rr_min_eff,
                rr_max=rr_max_eff,
                upsample_factor=upsample_factor,
            )
            pred_rr_windows.append(rr_pred)
            gt_rr_windows.append(rr_gt)

        # 每个样本取中位数，减小 outlier 影响
        all_pred_rr.append(float(np.median(pred_rr_windows)))
        all_gt_rr.append(float(np.median(gt_rr_windows)))

        if (idx + 1) % 5 == 0:
            print(f"[EvalFusion] Processed {idx+1}/{len(loader)} samples")

    # ---------------- 计算指标 ----------------
    all_pred_rr = np.array(all_pred_rr)
    all_gt_rr = np.array(all_gt_rr)

    mae = float(np.mean(np.abs(all_pred_rr - all_gt_rr)))
    rmse = float(np.sqrt(np.mean((all_pred_rr - all_gt_rr) ** 2)))
    if len(all_pred_rr) > 1:
        corr = float(np.corrcoef(all_pred_rr, all_gt_rr)[0, 1])
    else:
        corr = float("nan")

    t1 = time.time()

    print("============== EVAL FUSION (AUTO-MODEL) ==============")
    print(f"Model type     : {model_type}")
    print(f"Samples used   : {len(all_gt_rr)}")
    print(f"MAE   (bpm)    : {mae:.3f}")
    print(f"RMSE  (bpm)    : {rmse:.3f}")
    print(f"Corr(pred, gt) : {corr:.3f}")
    print(f"Time elapsed   : {t1 - t0:.2f} s")

    # 打印前几个样本的 GT / Pred 对
    print("First few pairs (GT | Pred_RR):")
    for i in range(min(10, len(all_gt_rr))):
        print(f"  #{i:02d}: {all_gt_rr[i]:6.2f} | {all_pred_rr[i]:6.2f}")


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="auto",
        help="checkpoint 路径；设为 auto 时自动从 checkpoints_fusion/ 里选最新 best_fusion_epoch_XXX.pth",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/dsr/OVRM/Motion_Dataset",
        help="数据集根目录（包含 1/2/3... 被试文件夹）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda / cpu / auto",
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=-1,
        help="评估用 clip 长度（帧）；<=0 时优先用 ckpt.cfg 里的 EVAL_CLIP_LEN/CLIP_LEN",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="滑动窗口步长（帧）；<=0 时优先用 ckpt.cfg 里的 EVAL_STEP/默认 200",
    )
    parser.add_argument(
        "--rr_min",
        type=float,
        default=-1.0,
        help="RR 最小 bpm；<=0 时优先用 ckpt.cfg 里的 EVAL_RR_MIN/MIN_BPM",
    )
    parser.add_argument(
        "--rr_max",
        type=float,
        default=-1.0,
        help="RR 最大 bpm；<=0 时优先用 ckpt.cfg 里的 EVAL_RR_MAX/MAX_BPM",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=UPSAMPLE_DEFAULT,
        help="compute_rr_from_resp 的 upsample_factor（频域零填充倍数）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 自动找 ckpt
    if args.ckpt == "auto":
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints_fusion_6")
        ckpt_path, ep = find_latest_best_ckpt(ckpt_dir)
        print(f"[EvalFusion] Auto-selected checkpoint: {ckpt_path} (epoch={ep})")
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
