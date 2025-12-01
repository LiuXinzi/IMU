import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from model_def import PoseLSTM

# try:
#     from fstar_watanabe import STAR  # Optional, used for edge extraction
# except ImportError:  # pragma: no cover - fallback when STAR is unavailable
#     STAR = None
from fstar_watanabe import STAR


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def denormalize(flat_array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean_flat = mean.reshape(-1)
    std_flat = std.reshape(-1)
    return flat_array * std_flat + mean_flat


# def get_smpl_edges() -> list:
#     if STAR is not None:
#         try:
#             star = STAR(gender="neutral", num_betas=10)
#             kintree = star.kintree_table
#             return [(int(kintree[0, i]), int(kintree[1, i])) for i in range(1, kintree.shape[1])]
#         except Exception:
#             pass

#     return [
#         (0, 1), (1, 2), (2, 3),
#         (0, 4), (4, 5), (5, 6),
#         (0, 7), (7, 8), (8, 9), (9, 10), (10, 11),
#         (8, 12), (12, 13), (13, 14), (14, 15),
#         (8, 16), (16, 17), (17, 18), (18, 19),
#         (9, 20), (20, 21), (14, 22), (22, 23),
#     ]

def get_smpl_edges() -> list:
    
    star = STAR(gender="neutral", num_betas=10, skip_forward=True)
    kintree = star.kintree_table
    return [(int(kintree[0, i]), int(kintree[1, i])) for i in range(1, kintree.shape[1])]
        


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
        # print(f"target{gt[0]}")
        # print(f"pred{pred[0]}")

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
                      joints_mean: np.ndarray, joints_std: np.ndarray,
                      center_index: int, device: torch.device,
                      chunk_size: int = 512) -> dict:
    """
    Evaluate a single sequence and compute error metrics.
    """
    model.eval()
    preds_chunks = []
    total = inputs.shape[0]

    with torch.no_grad():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            batch_inputs = inputs[start:end].to(device, non_blocking=True)
            _, joints_seq = model(batch_inputs)
            preds_chunks.append(joints_seq[:, center_index].cpu())

    preds = torch.cat(preds_chunks, dim=0).numpy()
    targets = joints_gt.cpu().numpy()

    # preds_denorm = denormalize(preds, joints_mean, joints_std)
    # targets_denorm = denormalize(targets, joints_mean, joints_std)

    num_joints = joints_mean.reshape(-1, 3).shape[0]
    # preds_denorm = preds_denorm.reshape(preds_denorm.shape[0], num_joints, 3)
    # targets_denorm = targets_denorm.reshape(targets_denorm.shape[0], num_joints, 3)

    preds = preds.reshape(preds.shape[0], num_joints, 3)
    targets = targets.reshape(targets.shape[0], num_joints, 3)

    errors = np.linalg.norm(preds - targets, axis=2)

    per_joint_mean = errors.mean(axis=0)
    per_joint_max = errors.max(axis=0)
    overall_mean = errors.mean()

    return {
        "num_frames": errors.shape[0],
        "per_joint_mean": per_joint_mean,
        "per_joint_max": per_joint_max,
        "overall_mean": overall_mean,
        "errors": errors,
        "preds": preds,
        "targets": targets,
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


    summary_path = Path("models_CMU/training_summary.json").resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    base_dir = summary_path.parent
    # model_key = "best_model_path"
    # model_path = resolve_path(base_dir, summary[model_key])
    model_path = resolve_path(base_dir, "best_model_rotate.pth")
    # norm_params_path = resolve_path(base_dir, summary["norm_params_path"])
    norm_params_path = resolve_path(base_dir, "norm_params_selected.npz")
    # test_data_path = resolve_path(base_dir, summary["train_data_path"])
    test_data_path = resolve_path(base_dir, summary["train_data_path"])
    center_index = int(summary.get("hyperparameters", {}).get("center_index", 20))

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
    center_index = int(test_pack.get("center_index", center_index))
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
        centers = seq.get("centers")
        centers_np = centers.cpu().numpy() if isinstance(centers, torch.Tensor) else None

        metrics = evaluate_sequence(model, inputs, joints, joints_mean, joints_std, center_index, device)
        metrics["centers"] = centers_np

        results.append({
            "file": seq["file"],
            "num_frames": metrics["num_frames"],
            "per_joint_mean": metrics["per_joint_mean"].tolist(),
            "per_joint_max": metrics["per_joint_max"].tolist(),
            "overall_mean": float(metrics["overall_mean"]),
            "centers": centers_np.tolist() if centers_np is not None else None,
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
        vis_dir = base_dir / "all_rot_selected_2"
        vis_dir.mkdir(parents=True, exist_ok=True)
        edges = get_smpl_edges()
        overall_means = np.array([metrics["overall_mean"] for metrics in sequence_metrics])
        sorted_indices = np.argsort(overall_means)
        select_k = min(22, len(sorted_indices))
        select_k1 = min(1, len(sorted_indices))
        visualization_payload = {"lowest": [], "highest": []}

        for rank, idx in enumerate(sorted_indices[:select_k], 1):
            metrics = sequence_metrics[idx]
            preds = metrics["preds"]
            targets = metrics["targets"]
            centers = metrics.get("centers")
            stem = Path(results[idx]["file"]).stem
            gif_path = vis_dir / f"{stem}_rank{rank}_lowest.gif"
            create_comparison_animation(preds, targets, edges, gif_path)
            visualization_payload["lowest"].append({
                "sequence_file": results[idx]["file"],
                "overall_mean": float(results[idx]["overall_mean"]),
                "gif_path": str(gif_path),
                "rank": rank,
                "centers": centers.tolist() if centers is not None else None,
            })
            print(f"Saved lowest error #{rank} animation to: {gif_path}")

        for rank, idx in enumerate(sorted_indices[::-1][:select_k1], 1):
            metrics = sequence_metrics[idx]
            preds = metrics["preds"]
            targets = metrics["targets"]
            centers = metrics.get("centers")
            stem = Path(results[idx]["file"]).stem
            gif_path = vis_dir / f"{stem}_rank{rank}_highest.gif"
            create_comparison_animation(preds, targets, edges, gif_path)
            visualization_payload["highest"].append({
                "sequence_file": results[idx]["file"],
                "overall_mean": float(results[idx]["overall_mean"]),
                "gif_path": str(gif_path),
                "rank": rank,
                "centers": centers.tolist() if centers is not None else None,
            })
            print(f"Saved highest error #{rank} animation to: {gif_path}")

        output_payload["visualizations"] = visualization_payload
        # print(preds.min(),preds.max())
        # print(len(edges))
        # print(edges[:5])

#     # -----------------------------------------
# # save top-100 highest error CMU IDs to txt
# # -----------------------------------------

# # extract file paths and their overall mean errors
#     file_errors = [(item["file"], item["overall_mean"]) for item in results]

# # sort errors from high to low
#     file_errors_sorted = sorted(file_errors, key=lambda x: x[1], reverse=True)

# # select top-100
#     top_k = min(100, len(file_errors_sorted))
#     top_100 = file_errors_sorted[:top_k]

# # extract CMU IDs from file paths (e.g., 16_34)
#     cmu_ids = [Path(f).stem for (f, _) in top_100]

#     # save to txt file
#     txt_path = base_dir / "evaluation" / "top100_highest_error_CMU_ids.txt"
#     txt_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(txt_path, "w") as f:
#         for cid in cmu_ids:
#            f.write(f"{cid}\n")

#     print(f"\nSaved top-100 highest error CMU IDs to: {txt_path}")




    # if args.output:
    #     output_path = Path(args.output).resolve()
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(output_path, "w", encoding="utf-8") as f:
    #         json.dump(output_payload, f, indent=2)
    #     print(f"Saved evaluation metrics to {output_path}")aaaaa


if __name__ == "__main__":
    main()
