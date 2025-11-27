# Data/motion_dataset.py
import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    """
    不依赖人脸检测的胸口 ROI 数据集（带基础数据清洗）。
    """

    def __init__(
        self,
        root_dir,
        resize=(128, 128),
        clip_len=300,
        transform=None,
        normalize_resp=True,
        split="train",
        val_ratio=0.2,
        min_video_frames=300,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.resize = resize
        self.clip_len = clip_len
        self.transform = transform
        self.normalize_resp = normalize_resp
        self.split = split
        self.val_ratio = val_ratio
        self.min_video_frames = min_video_frames

        all_items = self._scan_videos_and_resps()
        if len(all_items) == 0:
            raise RuntimeError(f"No valid (mp4 + npy) pairs found under {root_dir}")

        random.shuffle(all_items)
        n_total = len(all_items)
        n_val = int(n_total * val_ratio)

        if split == "train":
            self.items = all_items[n_val:]
        elif split == "val":
            self.items = all_items[:n_val]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        print(
            f"[MotionDataset] Split={split}, samples={len(self.items)} "
            f"(total={n_total})"
        )

    # ---------------- 数据清洗 ----------------
    @staticmethod
    def _check_resp_quality(
        resp: np.ndarray,
        rpath: str,
        min_len: int,
        min_std: float = 1e-4,
        max_abs: float = 1e4,
        max_nan_ratio: float = 0.1,
    ) -> bool:
        if resp.ndim == 0:
            print(f"[CLEAN] Resp is scalar, skip: {rpath}")
            return False

        length = len(resp)
        if length < min_len:
            print(
                f"[CLEAN] Resp signal too short ({length} < {min_len}), skip: {rpath}"
            )
            return False

        finite_mask = np.isfinite(resp)
        finite_ratio = float(finite_mask.mean())
        if finite_ratio < 1.0 - max_nan_ratio:
            print(
                f"[CLEAN] Too many NaN/Inf in resp (finite_ratio={finite_ratio:.2f}), "
                f"skip: {rpath}"
            )
            return False

        if not finite_mask.any():
            print(f"[CLEAN] Resp has no finite values, skip: {rpath}")
            return False

        resp_clean = resp[finite_mask]

        std = float(resp_clean.std())
        if std < min_std:
            print(
                f"[CLEAN] Resp std too small ({std:.3e}), almost constant, skip: {rpath}"
            )
            return False

        max_val = float(np.max(np.abs(resp_clean)))
        if max_val > max_abs:
            print(
                f"[CLEAN] Resp abs value too large (max={max_val:.3e}), "
                f"likely corrupted, skip: {rpath}"
            )
            return False

        return True

    def _scan_videos_and_resps(self):
        pairs = []
        for pid in sorted(os.listdir(self.root_dir)):
            pdir = os.path.join(self.root_dir, pid)
            if not os.path.isdir(pdir):
                continue
            for fname in os.listdir(pdir):
                if not fname.lower().endswith(".mp4"):
                    continue
                base = os.path.splitext(fname)[0]
                vpath = os.path.join(pdir, fname)
                rpath = os.path.join(pdir, base + ".npy")
                if not os.path.exists(rpath):
                    continue

                cap = cv2.VideoCapture(vpath)
                if not cap.isOpened():
                    print(f"[WARN] Cannot open video, skip: {vpath}")
                    cap.release()
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames < self.min_video_frames:
                    print(
                        f"[WARN] Video too short ({total_frames} frames), skip: {vpath}"
                    )
                    continue

                try:
                    resp = np.load(rpath).astype(np.float32)
                except Exception as e:
                    print(f"[CLEAN] Failed to load resp ({e}), skip: {rpath}")
                    continue

                len_resp = len(resp)

                if not self._check_resp_quality(
                    resp, rpath, min_len=self.min_video_frames
                ):
                    continue

                pairs.append(
                    {
                        "video_path": vpath,
                        "resp_path": rpath,
                        "num_frames": total_frames,
                        "resp_len": len_resp,
                    }
                )

        return pairs

    # ---------------- 胸口 ROI ----------------
    @staticmethod
    def _get_chest_box_geometric(
        H,
        W,
        box_h_ratio=0.4,
        box_w_ratio=0.4,
        center_y_ratio=0.6,
    ):
        h = int(H * box_h_ratio)
        w = int(W * box_w_ratio)

        cy = int(H * center_y_ratio)
        cx = W // 2

        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        if x2 <= x1 + 4:
            x2 = min(W, x1 + 4)
        if y2 <= y1 + 4:
            y2 = min(H, y1 + 4)

        return int(x1), int(y1), int(x2), int(y2)

    # ---------------- 采样 ----------------
    def __len__(self):
        return len(self.items)

    def _sample_clip_with_meta(self, idx, need_meta=False):
        item = self.items[idx]
        vpath = item["video_path"]
        rpath = item["resp_path"]
        num_frames = item["num_frames"]

        # 再清一次 NaN/Inf
        resp_all = np.load(rpath).astype(np.float32)
        finite_mask = np.isfinite(resp_all)
        if not finite_mask.all():
            if finite_mask.any():
                mean_val = float(resp_all[finite_mask].mean())
            else:
                mean_val = 0.0
            resp_all = resp_all.copy()
            resp_all[~finite_mask] = mean_val

        resp_len = len(resp_all)

        max_start = min(num_frames, resp_len) - self.clip_len
        if max_start <= 0:
            start = 0
        else:
            start = random.randint(0, max_start)

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vpath}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        used_indices = []
        chest_box = None

        for t in range(self.clip_len):
            ok, frm = cap.read()
            if not ok:
                break

            H, W = frm.shape[:2]
            if chest_box is None:
                chest_box = self._get_chest_box_geometric(H, W)

            x1, y1, x2, y2 = chest_box
            roi = frm[y1:y2, x1:x2]
            if roi is None or roi.size == 0:
                roi = frm.copy()

            roi = cv2.resize(roi, self.resize)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            frames.append(roi)
            used_indices.append(start + t)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames read from video: {vpath}")
        while len(frames) < self.clip_len:
            frames.append(frames[-1])
            used_indices.append(used_indices[-1] + 1)

        clip = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)           # [T,C,H,W]

        if self.transform is not None:
            for t in range(clip.size(0)):
                clip[t] = self.transform(clip[t])

        clip = clip.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]

        if start + self.clip_len <= resp_len:
            resp = resp_all[start:start + self.clip_len]
        else:
            resp = resp_all[start:]
            if len(resp) > 0:
                last_val = resp[-1]
                pad_len = self.clip_len - len(resp)
                resp = np.concatenate(
                    [resp, np.full(pad_len, last_val, dtype=np.float32)]
                )
            else:
                resp = np.zeros((self.clip_len,), dtype=np.float32)

        if self.normalize_resp:
            mean = resp.mean()
            std = resp.std()
            if std < 1e-6:
                std = 1e-6
            resp = (resp - mean) / std

        resp = torch.from_numpy(resp.astype(np.float32))

        if not need_meta:
            return clip, resp

        meta = {
            "video_path": vpath,
            "resp_path": rpath,
            "start": start,
            "used_frame_indices": used_indices,
            "roi_box": chest_box,
        }
        return clip, resp, meta

    def __getitem__(self, idx):
        clip, resp = self._sample_clip_with_meta(idx, need_meta=False)
        return clip, resp

    def get_item_with_meta(self, idx):
        return self._sample_clip_with_meta(idx, need_meta=True)
