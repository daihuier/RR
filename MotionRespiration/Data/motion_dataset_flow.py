# Data/motion_dataset_flow.py
# 完整可替换版本：在 MotionDataset 基础上加胸部 ROI + 简单光流 + 呼吸带通清洗
# 本版增加：
#   - clean_min_bpm / clean_max_bpm / resp_quality_thr 作为可配置参数
#   - 方便和训练 / eval 的 RR 频带统一调参

import os
import json
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from Data.motion_dataset import MotionDataset


# ============================================================
# 简单频域带通（按 bpm 设定上下限）
# ============================================================
def _bandpass_bpm(
    resp: np.ndarray,
    fs: float,
    min_bpm: float = 6.0,
    max_bpm: float = 40.0,
) -> np.ndarray:
    """
    只做一个简单的频域带通：
      - resp: 1D 呼吸信号，shape [T]
      - fs  : 采样频率 (Hz)，例如 30.0
      - min_bpm / max_bpm: 呼吸频率上下限（单位：bpm）

    返回：带通后的信号（float32）
    """
    resp = np.asarray(resp, dtype=np.float32)
    n = len(resp)

    # 太短就没法稳健做 FFT，直接减均值返回
    if n < 16:
        return resp - resp.mean()

    # 去掉直流分量
    resp_dc = resp - resp.mean()

    # 频率轴（只看 rfft 的非负频率）
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    F = np.fft.rfft(resp_dc)

    # bpm → Hz
    min_hz = float(min_bpm) / 60.0
    max_hz = float(max_bpm) / 60.0

    # 带通掩码：只保留 [min_hz, max_hz]
    band = (freqs >= min_hz) & (freqs <= max_hz)
    F[~band] = 0.0

    # 反变换回时域
    resp_filt = np.fft.irfft(F, n=n)
    return resp_filt.astype(np.float32)


# ============================================================
# 主类：MotionDatasetFlow
# ============================================================
class MotionDatasetFlow(MotionDataset):
    """
    在原有 MotionDataset 的基础上，额外生成一个“伪光流”并做更靠谱的预处理：

    - ROI：优先使用人脸框推胸部 ROI，缺省时退化为中间裁剪
    - RGB：ROI 裁剪 + 去亮度 + 标准化
    - Flow：基于 ROI 的 diff + grad_y，形状 [2, T, H, W]
    - resp：频域带通滤波 + 简单质量筛选（RR 不在合理范围内则视为低质量）

    __getitem__ 输出: (clip_rgb, flow, resp)
      clip_rgb: [3, T, H, W]
      flow    : [2, T, H, W]
      resp    : [T]
    """

    def __init__(
        self,
        root_dir: str,
        resize: Tuple[int, int] = (128, 128),
        clip_len: int = 300,
        normalize_resp: bool = False,
        split: str = "train",
        val_ratio: float = 0.2,
        fps: float = 30.0,
        face_json: Optional[str] = None,
        clean_min_bpm: float = 6.0,
        clean_max_bpm: float = 40.0,
        resp_quality_thr: float = 0.15,
    ):
        super().__init__(
            root_dir=root_dir,
            resize=resize,
            clip_len=clip_len,
            normalize_resp=normalize_resp,
            split=split,
            val_ratio=val_ratio,
        )

        self.resize = resize
        self.clip_len = clip_len
        self.split = split
        self.fps = float(fps)

        # 呼吸清洗相关超参数（方便从 cfg 里统一调）
        self.clean_min_bpm = float(clean_min_bpm)
        self.clean_max_bpm = float(clean_max_bpm)
        self.resp_quality_thr = float(resp_quality_thr)

        # ---------------- 加载人脸框 JSON ----------------
        if face_json is None:
            # 默认尝试项目根目录下的 face_boxes.json
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            face_json = os.path.join(project_root, "face_boxes.json")

        self.face_boxes = {}
        if face_json is not None and os.path.exists(face_json):
            try:
                with open(face_json, "r") as f:
                    self.face_boxes = json.load(f)
                print(f"[MotionDatasetFlow] Loaded face boxes from: {face_json}")
            except Exception as e:
                print(f"[MotionDatasetFlow][WARN] Failed to load face_json={face_json}: {e}")
        else:
            print(
                f"[MotionDatasetFlow] face_json not found, "
                f"will use center ROI only: {face_json}"
            )

        print(
            f"[MotionDatasetFlow] Init split='{self.split}', "
            f"root_dir='{root_dir}', resize={self.resize}, "
            f"clip_len={self.clip_len}, fps={self.fps}, "
            f"clean_band=[{self.clean_min_bpm},{self.clean_max_bpm}] bpm, "
            f"resp_quality_thr={self.resp_quality_thr}"
        )

    # ============================================================
    # 工具函数：根据 idx 尝试拿到视频文件名
    # ============================================================
    def _get_video_name(self, idx: int) -> str:
        """
        尽量从父类 MotionDataset 的样本列表里推断出视频文件名。
        如果拿不到，就返回一个占位名，这样只会走 fall-back ROI，不会报错。
        """
        name = None

        # 常见的属性名：self.samples / self.video_list / self.items / self.data_list
        cand_attrs = ["samples", "video_list", "items", "data_list"]
        for attr in cand_attrs:
            if hasattr(self, attr):
                lst = getattr(self, attr)
                if not (isinstance(lst, (list, tuple)) and 0 <= idx < len(lst)):
                    continue
                item = lst[idx]

                # dict: 尝试常见键
                if isinstance(item, dict):
                    for k in ["video_path", "path", "file", "filename"]:
                        if k in item:
                            name = os.path.basename(str(item[k]))
                            break

                # 字符串：直接当作路径
                elif isinstance(item, str):
                    name = os.path.basename(item)

                break

        if name is None:
            name = f"sample_{idx:04d}"
        return name

    # ============================================================
    # 根据人脸框推胸部 ROI（失败就用中心 ROI）
    # ============================================================
    def _get_chest_roi(self, video_name: str, H: int, W: int) -> Tuple[int, int, int, int]:
        """
        根据人脸框推断胸部 ROI：
        - 先在 face_boxes 里查 video_name 对应的 (fx1, fy1, fx2, fy2)
        - 胸部大致在脸框底部往下 0.6~2.0 个脸高的位置，左右扩一点
        如果查不到，就返回中间 ROI。
        """
        if video_name in self.face_boxes:
            try:
                fx1, fy1, fx2, fy2 = self.face_boxes[video_name]
                fw = fx2 - fx1
                fh = fy2 - fy1
                cx = 0.5 * (fx1 + fx2)

                ch_y1 = int(fy2 + 0.6 * fh)
                ch_y2 = int(fy2 + 2.0 * fh)
                ch_x1 = int(cx - 1.2 * fw)
                ch_x2 = int(cx + 1.2 * fw)

                # 边界裁剪
                ch_x1 = max(0, ch_x1)
                ch_y1 = max(0, ch_y1)
                ch_x2 = min(W, ch_x2)
                ch_y2 = min(H, ch_y2)

                # 防止出现高度/宽度为 0 的情况
                if ch_x2 - ch_x1 >= 8 and ch_y2 - ch_y1 >= 8:
                    return ch_x1, ch_y1, ch_x2, ch_y2
            except Exception as e:
                print(f"[MotionDatasetFlow][WARN] bad face box for {video_name}: {e}")

        # fallback：中间 ROI（大致覆盖胸部）
        y1 = int(H * 0.30)
        y2 = int(H * 0.85)
        x1 = int(W * 0.20)
        x2 = int(W * 0.80)
        return x1, y1, x2, y2

    # ============================================================
    # RGB 预处理：裁剪 ROI + 去亮度 + 标准化
    # ============================================================
    def _process_rgb_with_roi(self, frames: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        frames: [T, H, W, 3], float32 / uint8
        roi   : (x1, y1, x2, y2)
        返回：同样形状的裁剪 + 标准化后帧
        """
        T, H, W, C = frames.shape
        x1, y1, x2, y2 = roi

        # 边界安全处理
        x1 = max(0, min(W - 1, x1))
        x2 = max(x1 + 1, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(y1 + 1, min(H, y2))

        frames = frames[:, y1:y2, x1:x2, :]  # [T, H_roi, W_roi, 3]

        # 归一化到 [0,1]
        if frames.max() > 1.5:
            frames = frames.astype(np.float32) / 255.0
        else:
            frames = frames.astype(np.float32)

        # 每帧减掉自己的平均亮度
        frame_mean = frames.mean(axis=(1, 2, 3), keepdims=True)
        frames = frames - frame_mean

        # 整个 clip 做 per-channel 标准化
        flat = frames.reshape(-1, 3)
        std = flat.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        frames = frames / std.reshape(1, 1, 1, 3)

        return frames

    # ============================================================
    # 伪光流：diff + grad_y
    # ============================================================
    def _compute_flow_clip(self, frames: np.ndarray) -> np.ndarray:
        """
        输入:
            frames: [T, H, W, 3]，已经是 ROI 裁剪 + 标准化后的 RGB
        输出:
            flow: [2, T, H, W]，通道 0 为灰度帧差，通道 1 为灰度 y 向梯度
        """
        T, H, W, C = frames.shape

        # 灰度
        gray = (
            0.299 * frames[..., 0]
            + 0.587 * frames[..., 1]
            + 0.114 * frames[..., 2]
        )  # [T, H, W]

        # 时间帧差
        diff = np.zeros_like(gray)
        if T > 1:
            diff[1:] = gray[1:] - gray[:-1]

        # y 向梯度
        grad_y = np.zeros_like(gray)
        if H > 1:
            grad_y[:, 1:, :] = gray[:, 1:, :] - gray[:, :-1, :]

        # 标准化
        for arr in (diff, grad_y):
            s = arr.std()
            if s > 1e-6:
                arr /= s

        flow = np.stack([diff, grad_y], axis=0)  # [2, T, H, W]
        return flow.astype(np.float32)

    # ============================================================
    # resp 清洗 + 质量筛选
    # ============================================================
    def _clean_resp(self, resp_np: np.ndarray):
        """
        对原始呼吸信号做简单预处理：
        1) 长度检查（太短直接标无效）
        2) [clean_min_bpm, clean_max_bpm] 频段带通
        3) 按带通后能量 / 原始能量做一个粗糙质量评分

        返回:
            resp_clean: 预处理后的呼吸信号（np.float32, shape [T]）
            valid     : bool，质量是否可接受
        """
        resp = np.asarray(resp_np, dtype=np.float32)

        # ---- 1) 长度检查 ----
        if len(resp) < int(0.5 * self.clip_len):
            return resp, False

        # ---- 2) 频域带通 ----
        resp_bp = _bandpass_bpm(
            resp,
            fs=self.fps,
            min_bpm=self.clean_min_bpm,
            max_bpm=self.clean_max_bpm,
        )

        # ---- 3) 简单质量指标：带通后能量 / 原始能量 ----
        resp_raw_dc = resp - resp.mean()
        resp_bp_dc = resp_bp - resp_bp.mean()

        e_raw = float(np.mean(resp_raw_dc**2) + 1e-8)
        e_bp = float(np.mean(resp_bp_dc**2))

        quality = e_bp / e_raw  # “有用频段”占总能量的比例
        valid = quality > self.resp_quality_thr

        return resp_bp, valid

    # ============================================================
    # __getitem__
    # ============================================================
    def __getitem__(self, idx: int):
        """
        返回:
            clip_rgb: [3, T, H, W]
            flow    : [2, T, H, W]
            resp    : [T]
        """
        base = super().__getitem__(idx)

        # 兼容父类可能返回 (clip_rgb, resp) 或 (clip_rgb, resp, meta...)
        if isinstance(base, (list, tuple)):
            if len(base) >= 2:
                clip_rgb = base[0]
                resp = base[1]
            else:
                raise RuntimeError(
                    f"[MotionDatasetFlow] Unexpected parent __getitem__ length={len(base)}"
                )
        else:
            raise RuntimeError(
                "[MotionDatasetFlow] Parent __getitem__ must return tuple/list"
            )

        # clip_rgb: [3, T, H, W] (torch.Tensor)
        if not torch.is_tensor(clip_rgb):
            clip_rgb = torch.as_tensor(clip_rgb)
        if clip_rgb.dim() != 4:
            raise RuntimeError(
                f"[MotionDatasetFlow] clip_rgb.dim() must be 4, got {clip_rgb.dim()}"
            )

        # 拿到视频名（如果能拿到）
        video_name = self._get_video_name(idx)

        # [3, T, H, W] -> [T, H, W, 3]
        frames = clip_rgb.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]
        _, H_, W_, _ = frames.shape

        # ROI
        roi = self._get_chest_roi(video_name, H_, W_)
        frames_proc = self._process_rgb_with_roi(frames, roi)  # [T, H_roi, W_roi, 3]

        # Flow
        flow_np = self._compute_flow_clip(frames_proc)         # [2, T, H_roi, W_roi]

        # resp 清洗
        resp_np = (
            resp.cpu().numpy()
            if torch.is_tensor(resp)
            else np.asarray(resp, dtype=np.float32)
        )
        resp_clean, valid = self._clean_resp(resp_np)

        # 如果是训练集且样本质量很差，可以简单重采一个样本
        if (not valid) and (self.split == "train"):
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        # 转回 torch
        clip_rgb_t = (
            torch.from_numpy(frames_proc)
            .permute(3, 0, 1, 2)
            .float()
        )  # [3, T, H_roi, W_roi]
        flow_t = torch.from_numpy(flow_np).float()             # [2, T, H_roi, W_roi]
        resp_t = torch.from_numpy(resp_clean).float()          # [T]

        return clip_rgb_t, flow_t, resp_t
