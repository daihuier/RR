# train/train_fusion.py
import os
import torch

from train_fusion_core import run_training


class CFG:
    """
    主要改这里的配置：
    - 数据路径
    - batch size / lr / epoch 数
    - 呼吸频带、loss 配比、eval 滑窗 step 等

    支持通过 preset 快速切几组典型超参：
      - "baseline"
      - "narrow_band"
      - "wide_band_soft"
    """

    def __init__(self, preset: str = "baseline"):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.PROJECT_ROOT = project_root
        self.PRESET = preset

        # ---------- 数据 ----------
        self.DATA_DIR = "/data/dsr/OVRM/Motion_Dataset"

        # ---------- 设备 ----------
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- 时序参数 ----------
        # 注意：你的视频是 30 FPS，这里保持统一
        self.FPS = 30.0

        # 训练用 clip 长度（帧数）；建议和 eval 里的 EVAL_CLIP_LEN 保持一致
        # 600 帧 @ 30 FPS ≈ 20s
        self.CLIP_LEN = 600

        self.BATCH_SIZE = 4
        self.NUM_WORKERS = 4
        self.VAL_RATIO = 0.2

        # ---------- 模型结构开关 ----------
        # True : 使用 RGB + Flow 的 PhysNetFusion
        # False: 使用 RGB-only 的 PhysNet
        self.USE_FLOW = True

        # ---------- 训练超参数 ----------
        self.NUM_EPOCHS = 100
        self.LR = 1e-3
        self.WEIGHT_DECAY = 1e-4

        # ---------- 呼吸频带 & 清洗频带 ----------
        # 训练/验证内部使用的 RR 频带 (NMCC/AFD/RR_L1/FFT)
        self.MIN_BPM = 6.0
        self.MAX_BPM = 25.0   # 收窄到 25 以抑制倍频乱跳

        # 数据集内对 GT 呼吸做带通清洗时使用的频带
        # 一般可以略宽一点，避免过窄导致 GT 被误杀
        self.CLEAN_MIN_BPM = 6.0
        self.CLEAN_MAX_BPM = 40.0

        # 呼吸质量阈值（带通信号能量 / 原始能量）
        self.RESP_QUALITY_THR = 0.15

        # ---------- NMCC + AFD + RR + 高频 能量 Loss 权重 ----------
        # NMCC vs AFD 的权重 (这里稍微弱化 NMCC)
        self.NMCC_ALPHA = 0.4      # 原来 0.5 → 0.4

        # AFD/L1 的归一尺度, 单位 bpm
        self.AFD_SCALE = 10.0

        # RR_L1 的权重
        # baseline: 1.0，想更“频率优先”可以往上加
        self.LAMBDA_RR = 1.0

        # 高频谱能量惩罚权重 (压 30+ bpm 假峰)
        self.LAMBDA_HIGH = 1e-3

        # 高频惩罚阈值, 单位 bpm
        self.HIGH_BPM = 30.0

        # ---------- best 模型选择策略 ----------
        # score = mae_eval + BEST_COMBO_LAMBDA * max(0, BEST_TARGET_CORR - corr_eval)
        # 分数越小越好
        self.BEST_TARGET_CORR = 0.20     # 希望相关性至少达到的水平
        self.BEST_COMBO_LAMBDA = 3.0     # Corr 不达标时的惩罚系数 (越大越重视相关性)

        # ---------- Eval 相关超参（会写进 ckpt，eval_fusion 自动读取） ----------
        # 默认 eval clip 长度 = 训练 CLIP_LEN
        self.EVAL_CLIP_LEN = self.CLIP_LEN

        # 滑动窗口步长（帧）：
        #   - 想更平滑：步长小一些 (比如 100)
        #   - 想更快 / 更独立：步长大一些 (比如 200, 300)
        self.EVAL_STEP = 200

        # Eval 用的 RR 搜索频带（默认和训练频带保持一致）
        self.EVAL_RR_MIN = self.MIN_BPM
        self.EVAL_RR_MAX = 40

        # ---------- 杂项 ----------
        self.CHECKPOINT_DIR = os.path.join(project_root, "checkpoints_fusion")
        self.SEED = 42

        # ---------- 预设超参集（方便扫上限） ----------
        # 这里给你三组例子，你可以直接改数值或再加 preset
        preset_cfgs = {
            "baseline": {
                # 就用上面的默认值，不额外覆盖
            },
            "narrow_band": {
                # 频带收窄 + RR 项加重 + 高频惩罚更强
                "MIN_BPM": 8.0,
                "MAX_BPM": 22.0,
                "EVAL_RR_MIN": 8.0,
                "EVAL_RR_MAX": 22.0,
                "LAMBDA_RR": 1.5,
                "LAMBDA_HIGH": 2e-3,
            },
            "wide_band_soft": {
                # 频带放宽一点 + loss 更“软”，看结构极限
                "MIN_BPM": 4.0,
                "MAX_BPM": 30.0,
                "EVAL_RR_MIN": 4.0,
                "EVAL_RR_MAX": 30.0,
                "LAMBDA_RR": 0.7,
                "NMCC_ALPHA": 0.5,
                "LAMBDA_HIGH": 5e-4,
            },
        }

        if self.PRESET in preset_cfgs and preset_cfgs[self.PRESET]:
            print(f"[CFG] Applying preset: {self.PRESET}")
            for k, v in preset_cfgs[self.PRESET].items():
                setattr(self, k, v)
        else:
            if self.PRESET not in preset_cfgs:
                print(f"[CFG] Unknown preset '{self.PRESET}', fallback to baseline default.")

    # 你要的话可以加个 pretty_print，但现在不必须


def main():
    # 可以通过环境变量选择不同 preset：
    #   FUSION_PRESET=baseline / narrow_band / wide_band_soft
    preset = os.environ.get("FUSION_PRESET", "baseline")
    cfg = CFG(preset=preset)
    print(f"[CFG] Using preset: {cfg.PRESET}")
    run_training(cfg)


if __name__ == "__main__":
    main()
