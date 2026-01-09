import json
from pathlib import Path

import numpy as np
import torch

from model_def import PoseLSTM, PoseMLP

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("Motion/Running")          # folder with *.npz
MODEL_PATH = Path("lstm2/best_model.pth") # trained model checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PoseLSTM output dims (your setup)
INPUT_SIZE = 72
LEAF_OUT_SIZE = 18
FULL_OUT_SIZE = 72        # 24*3
NUM_JOINTS = 24
CHUNK_SIZE = 512


# -----------------------------
# I/O helpers (as you requested)
# -----------------------------
def load_npz(path: Path):
    data = np.load(path)
    if "x" not in data or "y" not in data:
        raise KeyError("npz 必须包含 x 和 y")
    x = data["x"]
    y = data["y"]
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return x, y


def to_joints(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2 and y.shape[1] == 72:
        return y.reshape(-1, 24, 3)
    if y.ndim == 3 and y.shape[1:] == (24, 3):
        return y
    raise ValueError(f"Unexpected y shape: {y.shape}")


# -----------------------------
# Evaluation
# -----------------------------
def predict_full_seq(model: PoseLSTM, x: np.ndarray) -> np.ndarray:
    """
    x: (T,72)
    return pred joints: (T,24,3)
    """
    model.eval()
    preds = []

    with torch.no_grad():
        T = x.shape[0]
        for start in range(0, T, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, T)
            xb = torch.from_numpy(x[start:end]).to(DEVICE)      # (t,72)
            xb = xb.unsqueeze(0)                                # (1,t,72)

            _, joints_seq = model(xb)                           # (1,t,72)
            pred_b = joints_seq[0].detach().cpu().numpy()        # (t,72)
            preds.append(pred_b)

    pred = np.concatenate(preds, axis=0)                         # (T,72)
    return to_joints(pred)



# def compute_metrics_over_dataset(model: PoseLSTM, npz_files: list[Path]) -> dict:
#     """
#     输出三个指标：
#       - overall_mean: 全数据集、全帧、全关节平均误差 (L2)
#       - overall_std:  全数据集、全帧、全关节误差标准差 (L2)
#       - overall_max:  全数据集、全帧、全关节最大误差 (L2)
#     """
#     # Welford online stats
#     n = 0
#     mean = 0.0
#     M2 = 0.0

#     overall_max = 0.0

#     for path in npz_files:
#         x, y = load_npz(path)             # x:(T,72) y:(T,72) or (T,24,3)
#         gt = to_joints(y)                 # (T,24,3)

#         pred = predict_full_seq(model, x) # (T,24,3)

#         T = min(pred.shape[0], gt.shape[0])
#         pred = pred[:T]
#         gt = gt[:T]

#         err = np.linalg.norm(pred - gt, axis=2)  # (T,24)

#         # update max
#         m = float(err.max())
#         if m > overall_max:
#             overall_max = m

#         # update mean/std with Welford
#         flat = err.reshape(-1)
#         for v in flat:
#             v = float(v)
#             n += 1
#             delta = v - mean
#             mean += delta / n
#             delta2 = v - mean
#             M2 += delta * delta2

#     overall_mean = mean
#     overall_var = (M2 / (n - 1)) if n > 1 else 0.0
#     overall_std = float(np.sqrt(overall_var))

#     return {
#         "overall_mean": float(overall_mean),
#         "overall_std": float(overall_std),
#         "overall_max": float(overall_max),
#         "num_files": len(npz_files),
#         "num_samples": int(n),  # 全数据集误差样本数 = sum(T*24)
#     }


def _third_derivative_from_positions(x: np.ndarray, dt: float, n: int = 4) -> np.ndarray:
    """
    用中心差分近似三阶导（jerk）:
      x'''(t) ≈ [x(t-2h) - 2x(t-h) + 2x(t+h) - x(t+2h)] / (2 h^3)
    where h = n*dt

    x: (T,24,3) 或 (T,*,3)
    return: (T-4n,24,3) 对应 t = 2n ... T-2n-1
    """
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"x must be (T,*,3), got {x.shape}")
    T = x.shape[0]
    if T <= 4 * n:
        return np.empty((0,) + x.shape[1:], dtype=np.float32)

    h = n * dt
    denom = 2.0 * (h ** 3)

    # 对齐：
    # t in [2n, T-2n-1], length = T - 4n
    jerk = (x[:-4*n] - 2.0 * x[n:-3*n] + 2.0 * x[3*n:-n] - x[4*n:]) / denom
    return jerk.astype(np.float32)


def compute_traj_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    keep_curve: bool = True,
    dt: float = 1/120,
    n: int = 4,
) -> dict:
    """
    单条轨迹指标（可直接 json.dump）。

    pred/gt: (T,24,3)

    输出：
      - overall_mean / overall_max (m)
      - std_over_frames (m)
      - jitter_mean (m/s^3) : 基于 pred 三阶导（jerk）的 RMS(|jerk|)
      - jitter_std  (m/s^3) : “逐帧 jitter(=每帧关节均值 jerk 模长)” 序列的 std
    """
    if pred.ndim != 3 or pred.shape[1:] != (24, 3):
        raise ValueError(f"pred shape must be (T,24,3), got {pred.shape}")
    if gt.ndim != 3 or gt.shape[1:] != (24, 3):
        raise ValueError(f"gt shape must be (T,24,3), got {gt.shape}")

    T = min(pred.shape[0], gt.shape[0])
    if T <= 0:
        out = {
            "T": 0,
            "overall_mean": 0.0,
            "overall_max": 0.0,
            "std_over_frames": 0.0,
            "jitter_mean": 0.0,
            "jitter_std": 0.0,
            "jitter_dt": float(dt),
            "jitter_n": int(n),
            "jitter_type": "third_derivative_jerk",
        }
        if keep_curve:
            out["frame_mean_err"] = []
            out["jitter_frame"] = []
        return out

    pred = pred[:T].astype(np.float32)
    gt   = gt[:T].astype(np.float32)

    # -------- error metrics (m) --------
    err = np.linalg.norm(pred - gt, axis=2).astype(np.float32)  # (T,24)
    overall_mean = float(err.mean())
    overall_max  = float(err.max())

    frame_mean_err = err.mean(axis=1)  # (T,)
    std_over_frames = float(frame_mean_err.std(ddof=1)) if T > 1 else 0.0

    # -------- jitter via 3rd derivative of pred (m/s^3) --------
    jerk = _third_derivative_from_positions(pred, dt=dt, n=n)  # (T-4n,24,3)

    if jerk.shape[0] == 0:
        jitter_mean = 0.0
        jitter_std = 0.0
        jitter_frame = np.empty((0,), dtype=np.float32)
    else:
        jerk_mag = np.linalg.norm(jerk, axis=2).astype(np.float32)  # (T-4n,24)

        # 1) 全局 jitter（RMS）
        jitter_mean = float(np.sqrt(np.mean(jerk_mag ** 2)))

        # 2) 逐帧 jitter 序列：每帧对关节取均值
        jitter_frame = jerk_mag.mean(axis=1)  # (T-4n,)

        # 3) jitter 的 std
        jitter_std = float(jitter_frame.std(ddof=1)) if jitter_frame.shape[0] > 1 else 0.0

    out = {
        "T": int(T),
        "overall_mean": overall_mean,
        "overall_max": overall_max,
        "std_over_frames": std_over_frames,
        "jitter_mean": float(jitter_mean),
        "jitter_std": float(jitter_std),
        "jitter_dt": float(dt),
        "jitter_n": int(n),
        "jitter_type": "third_derivative_jerk",
    }
    if keep_curve:
        out["frame_mean_err"] = frame_mean_err.astype(float).tolist()
        out["jitter_frame"] = jitter_frame.astype(float).tolist()
    return out





def compute_metrics_over_dataset(model: PoseLSTM, npz_files: list[Path]) -> dict:
    """
    数据集级指标 + 每条轨迹指标
    依赖你已有的：
      - load_npz(path) -> x,y
      - to_joints(y)   -> (T,24,3)
      - predict_full_seq(model,x) -> (T,24,3)
    """

    # 数据集整体（所有帧所有关节误差）在线统计：mean/std/max
    n = 0
    mean = 0.0
    M2 = 0.0
    dataset_overall_max = 0.0

    per_traj = []

    for path in npz_files:
        x, y = load_npz(path)
        gt = to_joints(y)                    # (T,24,3)
        pred = predict_full_seq(model, x)    # (T,24,3)

        # 轨迹级指标（含 std/jitter）
        tm = compute_traj_metrics(pred, gt)
        tm["file"] = str(path)
        per_traj.append(tm)

        # 更新数据集整体统计：基于该轨迹的每帧每关节误差
        T = tm["T"]
        pred = pred[:T]
        gt = gt[:T]
        err = np.linalg.norm(pred - gt, axis=2)  # (T,24)

        m = float(err.max())
        if m > dataset_overall_max:
            dataset_overall_max = m

        flat = err.reshape(-1)
        for v in flat:
            v = float(v)
            n += 1
            delta = v - mean
            mean += delta / n
            delta2 = v - mean
            M2 += delta * delta2

    dataset_overall_mean = float(mean)
    dataset_overall_var = (M2 / (n - 1)) if n > 1 else 0.0
    dataset_overall_std = float(np.sqrt(dataset_overall_var))

    # 轨迹级 std/jitter 的汇总（按轨迹平均）
    if len(per_traj) > 0:
        traj_std_mean = float(np.mean([d["jitter_std"] for d in per_traj]))
        traj_jitter_mean = float(np.mean([d["jitter_mean"] for d in per_traj]))
    else:
        traj_std_mean = 0.0
        traj_jitter_mean = 0.0

    return {
        "num_files": len(npz_files),
        "num_samples": int(n),  # sum(T*24)
        "dataset_overall_mean": dataset_overall_mean,
        "dataset_overall_std": dataset_overall_std,
        "dataset_overall_max": float(dataset_overall_max),
        "traj_std_mean": traj_std_mean,
        "traj_jitter_mean": traj_jitter_mean,
        "per_trajectory": per_traj,  # 如不想保存 frame_mean_err，可在 compute_traj_metrics 里去掉
    }


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz found in {DATA_DIR}")

    model = PoseLSTM(
        input_size=INPUT_SIZE,
        leaf_output_size=LEAF_OUT_SIZE,
        full_output_size=FULL_OUT_SIZE,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    metrics = compute_metrics_over_dataset(model, npz_files)

    # print(f"overall_mean_error (m): {metrics['overall_mean']:.6f}")
    # print(f"overall_max_joint_error_over_all_frames (m): {metrics['overall_max']:.6f}")

    out_path = Path("eval_summary.json").resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": str(DATA_DIR),
                "model_path": str(MODEL_PATH),
                **metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
