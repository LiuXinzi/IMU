import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from model_def import PoseLSTM

try:
    from fstar_watanabe import STAR  # Optional, used for edge extraction
except ImportError:  # pragma: no cover - fallback when STAR is unavailable
    STAR = None


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def denormalize(flat_array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean_flat = mean.reshape(-1)
    std_flat = std.reshape(-1)
    return flat_array * std_flat + mean_flat


def get_smpl_edges() -> list:
    if STAR is not None:
        try:
            star = STAR(gender="neutral", num_betas=10)
            kintree = star.kintree_table
            return [(int(kintree[0, i]), int(kintree[1, i])) for i in range(1, kintree.shape[1])]
        except Exception:
            pass

    return [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10), (10, 11),
        (8, 12), (12, 13), (13, 14), (14, 15),
        (8, 16), (16, 17), (17, 18), (18, 19),
        (9, 20), (20, 21), (14, 22), (22, 23),
    ]


def set_axes_equal(ax, coords: np.ndarray):
    coords = coords.reshape(-1, 3)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    if radius < 1e-6:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def create_comparison_animation(preds: np.ndarray, targets: np.ndarray, edges: list,
                                save_path: Path, fps: int = 15):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    all_coords = np.concatenate((preds.reshape(-1, 3), targets.reshape(-1, 3)), axis=0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Prediction vs Ground Truth")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    scatter_gt = ax.scatter([], [], [], c="tab:blue", s=25, label="GT")
    scatter_pred = ax.scatter([], [], [], c="tab:orange", s=25, label="Prediction")

    gt_lines = [ax.plot([], [], [], c="tab:blue", linewidth=1)[0] for _ in edges]
    pred_lines = [ax.plot([], [], [], c="tab:orange", linewidth=1, linestyle="--")[0] for _ in edges]

    set_axes_equal(ax, all_coords)
    ax.legend(loc="upper right")

    def update(frame_idx):
        gt = targets[frame_idx]
        pred = preds[frame_idx]

        scatter_gt._offsets3d = (gt[:, 0], gt[:, 1], gt[:, 2])
        scatter_pred._offsets3d = (pred[:, 0], pred[:, 1], pred[:, 2])

        for line, (parent, child) in zip(gt_lines, edges):
            line.set_data([gt[parent, 0], gt[child, 0]], [gt[parent, 1], gt[child, 1]])
            line.set_3d_properties([gt[parent, 2], gt[child, 2]])

        for line, (parent, child) in zip(pred_lines, edges):
            line.set_data([pred[parent, 0], pred[child, 0]], [pred[parent, 1], pred[child, 1]])
            line.set_3d_properties([pred[parent, 2], pred[child, 2]])

        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=preds.shape[0],
        interval=max(1, int(1000 / fps)),
        blit=False,
    )

    writer = animation.PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)


def evaluate_sequence(model: PoseLSTM, inputs: torch.Tensor, joints_gt: torch.Tensor,
                      joints_mean: np.ndarray, joints_std: np.ndarray, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        # import ipdb;ipdb.set_trace()
        _, joints_pred = model(inputs)
    preds = joints_pred.cpu().numpy()
    targets = joints_gt.numpy()

    preds_denorm = denormalize(preds, joints_mean, joints_std)
    targets_denorm = denormalize(targets, joints_mean, joints_std)

    num_joints = joints_mean.reshape(-1, 3).shape[0]
    preds_denorm = preds_denorm.reshape(preds_denorm.shape[0], num_joints, 3)
    targets_denorm = targets_denorm.reshape(targets_denorm.shape[0], num_joints, 3)

    errors = np.linalg.norm(preds_denorm - targets_denorm, axis=2)

    per_joint_mean = errors.mean(axis=0)
    per_joint_max = errors.max(axis=0)
    overall_mean = errors.mean()

    return {
        "num_frames": errors.shape[0],
        "per_joint_mean": per_joint_mean,
        "per_joint_max": per_joint_max,
        "overall_mean": overall_mean,
        "errors": errors,
        "preds": preds_denorm,
        "targets": targets_denorm,
    }


def aggregate_metrics(sequence_metrics: list) -> dict:
    if not sequence_metrics:
        raise ValueError("No sequence metrics to aggregate.")

    num_joints = sequence_metrics[0]["per_joint_mean"].shape[0]
    per_joint_sum = np.zeros(num_joints, dtype=np.float64)
    per_joint_max = np.full(num_joints, -np.inf, dtype=np.float64)
    total_frames = 0
    overall_sum = 0.0
    overall_count = 0

    for metrics in sequence_metrics:
        errors = metrics["errors"]
        per_joint_sum += errors.sum(axis=0)
        per_joint_max = np.maximum(per_joint_max, errors.max(axis=0))
        total_frames += errors.shape[0]
        overall_sum += errors.sum()
        overall_count += errors.size

    per_joint_mean = per_joint_sum / total_frames
    overall_mean = overall_sum / overall_count

    return {
        "per_joint_mean": per_joint_mean,
        "per_joint_max": per_joint_max,
        "overall_mean": overall_mean,
        "total_frames": total_frames,
    }


def main():


    summary_path = Path("models_KIT/training_summary.json").resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    base_dir = summary_path.parent
    model_key = "best_model_path"
    model_path = resolve_path(base_dir, summary[model_key])
    norm_params_path = resolve_path(base_dir, summary["norm_params_path"])
    test_data_path = resolve_path(base_dir, summary["test_data_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not norm_params_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_params_path}")
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    norm_data = np.load(norm_params_path)
    joints_mean = norm_data["joints_mean"].astype(np.float32)
    joints_std = norm_data["joints_std"].astype(np.float32)

    test_pack = torch.load(test_data_path)
    sequences = test_pack.get("sequences", [])
    if not sequences:
        raise RuntimeError("No test sequences available for evaluation.")
    # import ipdb;ipdb.set_trace()
    sample_inputs = sequences[0]["inputs"]
    sample_leaf = sequences[0]["leaf"]
    sample_joints = sequences[0]["joints"]

    input_size = sample_inputs.shape[-1]
    leaf_size = sample_leaf.shape[-1]
    joint_size = sample_joints.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseLSTM(
        input_size=input_size,
        leaf_output_size=leaf_size,
        full_output_size=joint_size,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results = []
    sequence_metrics = []

    for seq in sequences:
        inputs = seq["inputs"].float()
        joints = seq["joints"].float()

        metrics = evaluate_sequence(model, inputs, joints, joints_mean, joints_std, device)
        results.append({
            "file": seq["file"],
            "num_frames": metrics["num_frames"],
            "per_joint_mean": metrics["per_joint_mean"].tolist(),
            "per_joint_max": metrics["per_joint_max"].tolist(),
            "overall_mean": float(metrics["overall_mean"]),
        })
        sequence_metrics.append(metrics)

    aggregate = aggregate_metrics(sequence_metrics)
    aggregate_result = {
        "per_joint_mean": aggregate["per_joint_mean"].tolist(),
        "per_joint_max": aggregate["per_joint_max"].tolist(),
        "overall_mean": float(aggregate["overall_mean"]),
        "total_frames": int(aggregate["total_frames"]),
    }

   
    print("Per-sequence metrics:")
    for item in results:
        print(f"  {item['file']}: overall_mean={item['overall_mean']:.4f} m, frames={item['num_frames']}")
    print("Overall average error across all joints: "
          f"{aggregate_result['overall_mean']:.4f} (meters)")
    output_payload = {
        "model_path": str(model_path),
        "norm_params_path": str(norm_params_path),
        "test_data_path": str(test_data_path),
        "aggregate": aggregate_result,
        "per_sequence": results,
    }

    if sequence_metrics:
        worst_idx = int(np.argmin([item["overall_mean"] for item in results]))
        worst_metrics = sequence_metrics[worst_idx]
        preds = worst_metrics.pop("preds")
        targets = worst_metrics.pop("targets")
        edges = get_smpl_edges()

        vis_dir = base_dir / "evaluation"
        vis_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(results[worst_idx]["file"]).stem
        video_path = vis_dir / f"{stem}_comparison.gif"
        create_comparison_animation(preds, targets, edges, video_path)
        output_payload["visualization"] = {
            "sequence_file": results[worst_idx]["file"],
            "video_path": str(video_path),
        }
        print(f"Saved comparison animation to: {video_path}")

    # if args.output:
    #     output_path = Path(args.output).resolve()
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(output_path, "w", encoding="utf-8") as f:
    #         json.dump(output_payload, f, indent=2)
    #     print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
