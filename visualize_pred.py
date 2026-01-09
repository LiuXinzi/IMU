from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from model_def import PoseLSTM, PoseTransformer, PoseTransformerCond, PoseMLP

# ------------------- 运行配置（直接修改下方变量） ------------------- #
# 日本語メモ:
# - 推論して可視化するスクリプト。まずは NPZ_PATH / MODEL_PATH を合わせる。
# - VIS_MODE="compare" なら予測と正解を重ねて表示。
NPZ_PATH = Path("Motion/Running/02_03_poses.npz")          # 输入轨迹 npz，包含 x 和 y
MODEL_PATH = Path("mlp1/best_model.pth")
MODEL_TYPE = "MLP"           # "lstm"、"transformer"、"transformer_cond"
OUTPUT_PATH = Path("run_mlp.gif")            # .gif 或 .mp4
FPS = 30
CHUNK_LEN = 0                             # 0 表示整段推理；>0 会分段
VIS_MODE = "compare"                         # "pred" 或 "compare"

# ------------------- SMPL edges ------------------- #
# 日本語メモ:
# - (parent, child) で骨格の線を描く。
EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]

LEAF_IDS = [0, 7, 8, 12, 20, 21]  # 叶节点在 24 关节索引



def load_npz(path: Path):
    # 日本語メモ:
    # - npz から x/y を取り出し、float32 にそろえる。
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
    # 日本語メモ:
    # - 72次元 (24関節×3) を (24,3) に整形する。
    if y.ndim == 2 and y.shape[1] == 72:
        return y.reshape(-1, 24, 3)
    if y.ndim == 3 and y.shape[1:] == (24, 3):
        return y
    raise ValueError(f"Unexpected y shape: {y.shape}")


def pelvis_center(joints: np.ndarray) -> np.ndarray:
    # 日本語メモ:
    # - 骨盤(0番関節)を原点にして位置をそろえる。
    pelvis = joints[:, 0:1, :]
    return joints - pelvis


def build_model(device: torch.device):
    # 日本語メモ:
    # - MODEL_TYPE に応じてモデルを作って重みを読み込む。
    if MODEL_TYPE == "lstm":
        model = PoseLSTM(
            input_size=72,
            leaf_output_size=18,
            full_output_size=72,
        )
    elif MODEL_TYPE == "transformer":
        model = PoseTransformer(
            input_size=72,
            full_output_size=72,
        )
    elif MODEL_TYPE == "transformer_cond":
        model = PoseTransformerCond(
            input_size=72,
            leaf_output_size=18,
            full_output_size=72,
        )

    elif MODEL_TYPE == "MLP":
        model = PoseMLP(
            input_size=72,
            leaf_output_size=18,
            full_output_size=24 * 3,
        ).to(device)   

    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_full(model, x: np.ndarray, device: torch.device) -> np.ndarray:
    # 日本語メモ:
    # - 系列全体を一括で推論する。
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        if MODEL_TYPE == "transformer":
            full_pred = model(x_tensor)
        else:
            _, full_pred = model(x_tensor)
    return full_pred.squeeze(0).cpu().numpy()


def predict_chunked(model, x: np.ndarray, device: torch.device, chunk_len: int) -> np.ndarray:
    # 日本語メモ:
    # - 長い系列は分割して推論し、最後に結合する。
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_len):
            end = min(start + chunk_len, x.shape[0])
            x_seg = torch.from_numpy(x[start:end]).unsqueeze(0).to(device)
            if MODEL_TYPE == "transformer":
                full_pred = model(x_seg)
            else:
                _, full_pred = model(x_seg)
            preds.append(full_pred.squeeze(0).cpu().numpy())
    return np.concatenate(preds, axis=0)


def set_axes_equal(ax, coords: np.ndarray):
    # 日本語メモ:
    # - 3軸のスケールをそろえて歪みなく表示する。
    coords = coords.reshape(-1, 3)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def animate(joints_pred: np.ndarray, joints_gt: np.ndarray | None, save_path: Path, fps: int):
    # 日本語メモ:
    # - 3Dの関節と骨格線をアニメーションにして保存する。
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Pose Prediction")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter_pred = ax.scatter([], [], [], c="tab:orange", s=20, label="Pred")
    pred_lines = [ax.plot([], [], [], c="tab:orange", linewidth=1)[0] for _ in EDGES]

    scatter_gt = None
    gt_lines = None
    if joints_gt is not None:
        scatter_gt = ax.scatter([], [], [], c="tab:blue", s=20, label="GT")
        gt_lines = [ax.plot([], [], [], c="tab:blue", linewidth=1, linestyle="--")[0] for _ in EDGES]

    all_coords = joints_pred
    if joints_gt is not None:
        all_coords = np.concatenate([joints_pred, joints_gt], axis=0)
    set_axes_equal(ax, all_coords)
    ax.legend(loc="upper right")

    def update(frame):
        # 日本語メモ:
        # - 各フレームで点と線の位置を更新。
        pred = joints_pred[frame]
        scatter_pred._offsets3d = (pred[:, 0], pred[:, 1], pred[:, 2])
        for line, (p, c) in zip(pred_lines, EDGES):
            line.set_data([pred[p, 0], pred[c, 0]], [pred[p, 1], pred[c, 1]])
            line.set_3d_properties([pred[p, 2], pred[c, 2]])

        if joints_gt is not None and scatter_gt is not None and gt_lines is not None:
            gt = joints_gt[frame]
            scatter_gt._offsets3d = (gt[:, 0], gt[:, 1], gt[:, 2])
            for line, (p, c) in zip(gt_lines, EDGES):
                line.set_data([gt[p, 0], gt[c, 0]], [gt[p, 1], gt[c, 1]])
                line.set_3d_properties([gt[p, 2], gt[c, 2]])
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=joints_pred.shape[0],
        interval=max(1, int(1000 / fps)),
        blit=False,
    )

    if save_path.suffix.lower() == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=writer)
    plt.close(fig)


def main():
    # 日本語メモ:
    # - 入力/モデルを読み込み → 推論 → 表示/保存。
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"npz not found: {NPZ_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")

    x, y = load_npz(NPZ_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    if CHUNK_LEN and CHUNK_LEN > 0:
        y_pred = predict_chunked(model, x, device, CHUNK_LEN)
    else:
        y_pred = predict_full(model, x, device)

    joints_pred = pelvis_center(to_joints(y_pred))
    joints_gt = pelvis_center(to_joints(y)) if VIS_MODE == "compare" else None

    animate(joints_pred, joints_gt, OUTPUT_PATH, FPS)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
