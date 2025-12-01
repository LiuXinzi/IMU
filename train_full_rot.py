import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb

from model_def import PoseLSTM


# ------------------- hyper parameters ------------------- #
TRAIN_RATIO = 0.8          
VAL_RATIO = 0.15           
NUM_PAST_FRAME = 30
NUM_FUTURE_FRAME = 5
WINDOW_SIZE = NUM_PAST_FRAME + NUM_FUTURE_FRAME + 1
CENTER_INDEX = NUM_PAST_FRAME
STEP_SIZE_TRAIN = 1
STEP_SIZE_EVAL = 1
BATCH_SIZE = 1            # Batch size must be 1 for full sequence training
EPOCHS = 50
LEARNING_RATE = 3e-4
L2_LAMBDA = 7e-5
RANDOM_SEED = 42


# ------------------- WandB Initialize ------------------- #
wandb.init(
    project="Pose_LSTM",
    name="PoseLSTM_rotation_input",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "window_size": WINDOW_SIZE,
        "center_index": CENTER_INDEX,
        "step_size_train": STEP_SIZE_TRAIN,
        "step_size_eval": STEP_SIZE_EVAL,
        "model": "PoseLSTM-two-stage",
        "optimizer": "AdamW",
        "loss": "MSE(leaf) + MSE(full)",
        "seed": RANDOM_SEED,
        "l2_lambda": L2_LAMBDA,
    },
)


# ------------------- Dataset & collate ------------------- #
class FullSeqDataset(torch.utils.data.Dataset):
    """
    All sequence dataset for full sequence training:
      inputs: (T, input_dim)
      leaf:   (T, leaf_dim)
      joints: (T, joint_dim)
    """
    def __init__(self, inputs_list, leaf_list, joint_list):
        self.inputs_list = inputs_list
        self.leaf_list = leaf_list
        self.joint_list = joint_list

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs_list[idx], dtype=torch.float32),
            torch.tensor(self.leaf_list[idx], dtype=torch.float32),
            torch.tensor(self.joint_list[idx], dtype=torch.float32),
        )





# ------------------- Data IO & preprocessing ------------------- #
def load_split(file_list):
    acc_list, quat_list, joints_list, leaf_list = [], [], [], []
    for path in file_list:
        data = np.load(path)
        acc_list.append(data["acc"].astype(np.float32))        # (T, 6, 3)
        quat_list.append(data["quat"].astype(np.float32))      # (T, 6, 3, 3)  ← 旋转矩阵
        joints_list.append(data["joints"].astype(np.float32))  # (T, J, 3)
        leaf_list.append(data["leaf_pos"].astype(np.float32))  # (T, 6, 3)
    return acc_list, quat_list, joints_list, leaf_list


def compute_norm_stats(data_list):
    """
    Compute mean and std for a list of sequences (flattened to (N_total, D)).
    """
    if not data_list:
        raise ValueError("Empty data list when computing normalization statistics.")
    feature_shape = data_list[0].shape[1:]
    flat = np.concatenate(
        [arr.reshape(arr.shape[0], -1) for arr in data_list],
        axis=0,
    )
    mean = flat.mean(axis=0).reshape(feature_shape).astype(np.float32)
    std = (flat.std(axis=0) + 1e-6).reshape(feature_shape).astype(np.float32)
    return mean, std


def normalize_list(data_list, mean, std, apply=True):
    """
    Normalize a list of sequences with given mean and std. if apply is False, return original data.
    """
    if not apply:
        return data_list

    flat_mean = mean.reshape(1, -1)
    flat_std = std.reshape(1, -1)
    normalized = []
    for arr in data_list:
        shape = arr.shape
        flat = arr.reshape(shape[0], -1)
        norm_flat = (flat - flat_mean) / flat_std
        normalized.append(norm_flat.reshape(shape))
    return normalized


def flatten_sequences(seq_list):
    """
    (T, ..., D) → (T, D_all)
    For example, joints: (T, J, 3) → (T, 3J)
    """
    return [seq.reshape(seq.shape[0], -1) for seq in seq_list]


def create_windows(inputs, leaf, joints, window_size, center_idx, step_size, return_centers=False):
    """
    Generate sliding windows from a single sequence (used for saving test/train_sequences).
    inputs: (T, Din)
    leaf:   (T, Dleaf)
    joints: (T, Djoint)
    """
    if inputs.shape[0] < window_size:
        empty_inputs = np.empty((0, window_size, inputs.shape[1]), dtype=np.float32)
        empty_leaf = np.empty((0, leaf.shape[1]), dtype=np.float32)
        empty_joints = np.empty((0, joints.shape[1]), dtype=np.float32)
        if return_centers:
            return empty_inputs, empty_leaf, empty_joints, np.empty((0,), dtype=np.int64)
        return empty_inputs, empty_leaf, empty_joints

    windows, leaf_targets, joint_targets, centers = [], [], [], []
    for start in range(0, inputs.shape[0] - window_size + 1, step_size):
        end = start + window_size
        center = start + center_idx
        windows.append(inputs[start:end])
        leaf_targets.append(leaf[center])
        joint_targets.append(joints[center])
        centers.append(center)

    windows = np.stack(windows).astype(np.float32)
    leaf_targets = np.stack(leaf_targets).astype(np.float32)
    joint_targets = np.stack(joint_targets).astype(np.float32)
    centers = np.asarray(centers, dtype=np.int64)

    if return_centers:
        return windows, leaf_targets, joint_targets, centers
    return windows, leaf_targets, joint_targets


def create_windows_from_lists(inputs_list, leaf_list, joints_list, window_size, center_idx, step_size):
    all_inputs, all_leaf, all_joints = [], [], []
    for inputs, leaf, joints in zip(inputs_list, leaf_list, joints_list):
        windows, leaf_targets, joint_targets = create_windows(
            inputs, leaf, joints, window_size, center_idx, step_size
        )
        if windows.shape[0] == 0:
            continue
        all_inputs.append(windows)
        all_leaf.append(leaf_targets)
        all_joints.append(joint_targets)

    if not all_inputs:
        feature_dim = inputs_list[0].shape[1]
        leaf_dim = leaf_list[0].shape[1]
        joint_dim = joints_list[0].shape[1]
        return (
            np.empty((0, window_size, feature_dim), dtype=np.float32),
            np.empty((0, leaf_dim), dtype=np.float32),
            np.empty((0, joint_dim), dtype=np.float32),
        )

    return (
        np.concatenate(all_inputs, axis=0),
        np.concatenate(all_leaf, axis=0),
        np.concatenate(all_joints, axis=0),
    )


def build_inputs(acc_list, quat_list):
    """
    把 acc (T,6,3) 和 rot-mat quat (T,6,3,3) 拼成一帧的特征向量：
      acc_flat:  (T, 6, 3)
      rot_flat:  (T, 6, 9)
      combined:  (T, 6, 12) → reshape 为 (T, 72)
    """
    inputs = []
    for acc, quat in zip(acc_list, quat_list):
        # acc:  (T, 6, 3)
        # quat: (T, 6, 3, 3)  rotation matrix nor quaternion
        T, N, _ = acc.shape
        rot_flat = quat.reshape(T, N, 9)                     # (T, 6, 9)
        combined = np.concatenate([acc, rot_flat], axis=-1)  # (T, 6, 12)
        inputs.append(combined.reshape(T, -1).astype(np.float32))  # (T, 72)
    return inputs


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------- main ------------------- #
def main():
    set_seed(RANDOM_SEED)

    base_dir = Path(__file__).resolve().parent
    processed_dir = base_dir / "CMU_selected_process"   # Your current data directory
    models_dir = base_dir / "models_CMU"
    test_data_dir = models_dir / "test_data"

    models_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(processed_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")

    rng = np.random.default_rng(RANDOM_SEED)
    files = list(files)
    rng.shuffle(files)

    n_files = len(files)
    n_train = max(1, int(n_files * TRAIN_RATIO))

    train_files = files[:n_train]
    n_train = max(1, int(n_files * TRAIN_RATIO))
    n_val = max(1, int(n_files * VAL_RATIO))
    n_test = max(1, n_files - n_train - n_val)
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]   

    print(f"train_file: {len(train_files)}")

    # --------- Load raw data --------- #
    acc_tr, quat_tr, joints_tr, leaf_tr = load_split(train_files)
    acc_val, quat_val, joints_val, leaf_val = load_split(val_files)
    acc_te, quat_te, joints_te, leaf_te = load_split(test_files)

    # --------- Compute statistics (only used for acc/leaf/joints) --------- #
    acc_mean, acc_std = compute_norm_stats(acc_tr)
    quat_mean, quat_std = compute_norm_stats(quat_tr)    # Although computed, may not be used
    leaf_mean, leaf_std = compute_norm_stats(leaf_tr)
    joints_mean, joints_std = compute_norm_stats(joints_tr)

    # # --------- Normalize: only for acc / leaf / joints --------- #
    # acc_tr = normalize_list(acc_tr, acc_mean, acc_std, apply=True)
    # acc_te = normalize_list(acc_te, acc_mean, acc_std, apply=True)

    ACC_SCALE = 20.0  # Or 30, 50, depending on your data range

    acc_tr = [acc / ACC_SCALE for acc in acc_tr]
    acc_val = [acc / ACC_SCALE for acc in acc_val]
    acc_te = [acc / ACC_SCALE for acc in acc_te]
    # print(acc_tr)

# Do not save acc_mean / acc_std


    # Rotation matrices are not normalized to preserve original [-1,1] orthogonality
    quat_tr = normalize_list(quat_tr, quat_mean, quat_std, apply=False)
    quat_val = normalize_list(quat_val, quat_mean, quat_std, apply=False)
    quat_te = normalize_list(quat_te, quat_mean, quat_std, apply=False)

    leaf_tr = normalize_list(leaf_tr, leaf_mean, leaf_std, apply=False)
    leaf_te = normalize_list(leaf_te, leaf_mean, leaf_std, apply=False)

    joints_tr = normalize_list(joints_tr, joints_mean, joints_std, apply=False)
    joints_te = normalize_list(joints_te, joints_mean, joints_std, apply=False)

    # --------- 构建网络输入 & 展平 label --------- #
    inputs_tr = build_inputs(acc_tr, quat_tr)
    inputs_val = build_inputs(acc_val, quat_val)
    inputs_te = build_inputs(acc_te, quat_te)

    leaf_tr_flat = flatten_sequences(leaf_tr)
    leaf_val_flat = flatten_sequences(leaf_val)
    leaf_te_flat = flatten_sequences(leaf_te)

    joints_tr_flat = flatten_sequences(joints_tr)
    joints_val_flat = flatten_sequences(joints_val)
    joints_te_flat = flatten_sequences(joints_te)

    # --------- 训练集：全序列 Dataset --------- #
    train_dataset = FullSeqDataset(inputs_tr, leaf_tr_flat, joints_tr_flat)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_dataset = FullSeqDataset(inputs_val, leaf_val_flat, joints_val_flat)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # --------- save as the windows for evaluation --------- #
    X_test_list, leaf_test_list, joints_test_list, center_test_list = [], [], [], []
    for inputs, leaf, joints in zip(inputs_te, leaf_te_flat, joints_te_flat):
        X_w, leaf_w, joints_w, centers = create_windows(
            inputs, leaf, joints, WINDOW_SIZE, CENTER_INDEX, STEP_SIZE_EVAL, return_centers=True
        )
        X_test_list.append(X_w)
        leaf_test_list.append(leaf_w)
        joints_test_list.append(joints_w)
        center_test_list.append(centers)

    # --------- Build model --------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = inputs_tr[0].shape[-1]
    leaf_size = leaf_tr_flat[0].shape[-1]
    joint_size = joints_tr_flat[0].shape[-1]
    print(f"joint{joint_size}")

    model = PoseLSTM(
        input_size=input_size,
        leaf_output_size=leaf_size,
        full_output_size=joint_size,
    ).to(device)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)

    best_val = None
    best_epoch = None
    best_model_path = models_dir / "best_model_rotate.pth"

    # ------------------- Training Loop ------------------- #
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_leaf_loss = 0.0
        train_full_loss = 0.0
        train_samples = 0

        for X_batch, leaf_batch, joint_batch in train_loader:
            X_batch = X_batch.to(device)
            leaf_batch = leaf_batch.to(device)
            joint_batch = joint_batch.to(device)

            optimizer.zero_grad()
            leaf_seq, full_seq = model(X_batch)

            # MSE per element
            leaf_loss = criterion(leaf_seq, leaf_batch)       # (B,T,Dleaf)
            full_loss = criterion(full_seq, joint_batch)      # (B,T,Djoint)

            
            
            

            loss = leaf_loss + full_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_leaf_loss += leaf_loss.item()
            train_full_loss += full_loss.item()
            train_samples += leaf_batch.size(0)

        train_leaf_loss /= max(1, train_samples)
        train_full_loss /= max(1, train_samples)
        train_total_loss = train_leaf_loss + train_full_loss

        val_leaf_loss = None
        val_full_loss = None
        val_total_loss = None

        if val_loader is not None:
            model.eval()
            val_leaf_loss = 0.0
            val_full_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for X_batch, leaf_batch, joint_batch in val_loader:
                    X_batch = X_batch.to(device)
                    leaf_batch = leaf_batch.to(device)
                    joint_batch = joint_batch.to(device)
                    leaf_seq, full_seq = model(X_batch)
                    # leaf_pred = leaf_seq[:, CENTER_INDEX]
                    # full_pred = full_seq[:, CENTER_INDEX]
                    mse_leaf = criterion(leaf_seq, leaf_batch)
                    mse_full = criterion(full_seq, joint_batch)
                    mask_leaf = (leaf_batch != 0).float()
                    leaf_loss = (mse_leaf * mask_leaf).sum() / mask_leaf.sum()
                    mask_full = (joint_batch != 0).float()
                    full_loss = (mse_full * mask_full).sum() / mask_full.sum()
                    val_leaf_loss += leaf_loss.item()
                    val_full_loss += full_loss.item()
                    val_samples += leaf_batch.size(0)

            val_leaf_loss /= val_samples
            val_full_loss /= val_samples
            val_total_loss = val_leaf_loss + val_full_loss

            if best_val is None or val_total_loss < best_val:
                best_val = val_total_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                wandb.log({
                    "best_epoch": best_epoch,
                    "best_val_total_loss": best_val,
                    "best_val_leaf_loss": val_leaf_loss,
                    "best_val_full_loss": val_full_loss,
                })


        epoch_time = time.time() - epoch_start

        log_dict = {
            "epoch": epoch + 1,
            "train_leaf_loss": train_leaf_loss,
            "train_full_loss": train_full_loss,
            "train_total_loss": train_total_loss,
            "epoch_time_s": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        }
        wandb.log(log_dict)

        if val_total_loss is not None:
            log_dict.update({
                "val_leaf_loss": val_leaf_loss,
                "val_full_loss": val_full_loss,
                "val_total_loss": val_total_loss,
            })
        wandb.log(log_dict)

        msg = (
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"TrainLeaf: {train_leaf_loss:.4f}  "
            f"TrainFull: {train_full_loss:.4f}  "
            f"TrainTotal: {train_total_loss:.4f}  "
            f"Time: {epoch_time:.2f}s"
        )
        print(msg)

        if val_total_loss is not None:
            msg += (
                f"ValLeaf: {val_leaf_loss:.4f}  "
                f"ValFull: {val_full_loss:.4f}  "
                f"ValTotal: {val_total_loss:.4f}  "
            )
        msg += f"Time: {epoch_time:.2f}s"
        print(msg)

    # ------------------- Save model & stats ------------------- #
    last_model_path = models_dir / "last_model_selected.pth"
    torch.save(model.state_dict(), last_model_path)

    norm_params_path = models_dir / "norm_params_selected.npz"
    np.savez(
        norm_params_path,
        acc_mean=acc_mean,
        acc_std=acc_std,
        quat_mean=quat_mean,      # Although not used for normalization, saved for reference
        quat_std=quat_std,
        leaf_mean=leaf_mean,
        leaf_std=leaf_std,
        joints_mean=joints_mean,
        joints_std=joints_std,
    )

    # save test sequences (windowed version)
    test_sequences = []
    for path, X_w, leaf_w, joints_w, centers in zip(
        test_files, X_test_list, leaf_test_list, joints_test_list, center_test_list
    ):
        if X_w.size == 0:
            continue
        test_sequences.append({
            "file": str(path),
            "inputs": torch.from_numpy(X_w),
            "leaf": torch.from_numpy(leaf_w),
            "joints": torch.from_numpy(joints_w),
            "centers": torch.from_numpy(centers),
        })

    # save train sequences (windowed version)
    train_sequences = []
    for path, inputs, leaf, joints in zip(train_files, inputs_tr, leaf_tr_flat, joints_tr_flat):
        X_w, leaf_w, joints_w, centers = create_windows(
            inputs, leaf, joints, WINDOW_SIZE, CENTER_INDEX, STEP_SIZE_EVAL, return_centers=True
        )
        if X_w.size == 0:
            continue
        train_sequences.append({
            "file": str(path),
            "inputs": torch.from_numpy(X_w),
            "leaf": torch.from_numpy(leaf_w),
            "joints": torch.from_numpy(joints_w),
            "centers": torch.from_numpy(centers),
        })

    test_data_path = test_data_dir / "test_sequences.pt"
    torch.save({
        "window_size": WINDOW_SIZE,
        "center_index": CENTER_INDEX,
        "step_size": STEP_SIZE_EVAL,
        "sequences": test_sequences,
    }, test_data_path)

    train_data_path = test_data_dir / "train_sequences.pt"
    torch.save({
        "window_size": WINDOW_SIZE,
        "center_index": CENTER_INDEX,
        "step_size": STEP_SIZE_EVAL,
        "sequences": train_sequences,
    }, train_data_path)

    summary_path = models_dir / "training_summary.json"
    summary = {
        "train_files": [str(p) for p in train_files],
        "test_files": [str(p) for p in test_files],
        "last_model_path": str(last_model_path),
        "norm_params_path": str(norm_params_path),
        "test_data_path": str(test_data_path),
        "train_data_path": str(train_data_path),
        "hyperparameters": {
            "window_size": WINDOW_SIZE,
            "center_index": CENTER_INDEX,
            "step_size_train": STEP_SIZE_TRAIN,
            "step_size_eval": STEP_SIZE_EVAL,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "l2_lambda": L2_LAMBDA,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best model (if saved manually) path: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")
    print(f"Normalization parameters saved to: {norm_params_path}")
    print(f"Test sequences saved to: {test_data_path}")
    print(f"Train sequences saved to: {train_data_path}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
