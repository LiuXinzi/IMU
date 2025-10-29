import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb

from model_def import PoseLSTM


# ------------------- 超参数 ------------------- #
TRAIN_RATIO = 0.85
VAL_RATIO = 0.1
WINDOW_SIZE = 30
STEP_SIZE_TRAIN = 5
STEP_SIZE_EVAL = 1
BATCH_SIZE = 1024
EPOCHS = 2000
LEARNING_RATE = 1e-4
RANDOM_SEED = 42


# ------------------- WandB 初始化 ------------------- #
wandb.init(
    project="Pose_LSTM",
    name="PoseLSTM_two_stage",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "window_size": WINDOW_SIZE,
        "step_size_train": STEP_SIZE_TRAIN,
        "step_size_eval": STEP_SIZE_EVAL,
        "model": "PoseLSTM-two-stage",
        "optimizer": "Adam",
        "loss": "MSE(leaf) + MSE(full)",
        "seed": RANDOM_SEED,
    },
)


def load_split(file_list):
    acc_list, quat_list, joints_list, leaf_list = [], [], [], []
    for path in file_list:
        data = np.load(path)
        acc_list.append(data["acc"].astype(np.float32))
        quat_list.append(data["quat"].astype(np.float32))
        joints_list.append(data["joints"].astype(np.float32))
        leaf_list.append(data["leaf_pos"].astype(np.float32))
    return acc_list, quat_list, joints_list, leaf_list


def compute_norm_stats(data_list):
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


def normalize_list(data_list, mean, std):
    flat_mean = mean.reshape(1, -1)
    flat_std = std.reshape(1, -1)
    normalized = []
    for arr in data_list:
        shape = arr.shape
        flat = arr.reshape(shape[0], -1)
        norm_flat = (flat - flat_mean) / flat_std
        normalized.append(norm_flat.reshape(shape))
    return normalized


def create_windows(inputs, leaf, joints, window_size, step_size):
    if inputs.shape[0] < window_size:
        return np.empty((0, window_size, inputs.shape[1] * inputs.shape[2]), dtype=np.float32), \
               np.empty((0, leaf.shape[1] * leaf.shape[2]), dtype=np.float32), \
               np.empty((0, joints.shape[1] * joints.shape[2]), dtype=np.float32)

    feature_dim = int(np.prod(inputs.shape[1:]))
    leaf_dim = int(np.prod(leaf.shape[1:]))
    joint_dim = int(np.prod(joints.shape[1:]))

    inputs_flat = inputs.reshape(inputs.shape[0], feature_dim)
    leaf_flat = leaf.reshape(leaf.shape[0], leaf_dim)
    joints_flat = joints.reshape(joints.shape[0], joint_dim)

    X_windows, leaf_targets, joint_targets = [], [], []
    for start in range(0, inputs.shape[0] - window_size + 1, step_size):
        end = start + window_size
        X_windows.append(inputs_flat[start:end])
        leaf_targets.append(leaf_flat[end - 1])
        joint_targets.append(joints_flat[end - 1])

    return (
        np.stack(X_windows).astype(np.float32),
        np.stack(leaf_targets).astype(np.float32),
        np.stack(joint_targets).astype(np.float32),
    )


def create_windows_from_lists(inputs_list, leaf_list, joints_list, window_size, step_size):
    all_inputs, all_leaf, all_joints = [], [], []
    for inputs, leaf, joints in zip(inputs_list, leaf_list, joints_list):
        X_w, leaf_w, joints_w = create_windows(inputs, leaf, joints, window_size, step_size)
        if X_w.size == 0:
            continue
        all_inputs.append(X_w)
        all_leaf.append(leaf_w)
        all_joints.append(joints_w)

    if not all_inputs:
        return (
            np.empty((0, window_size, inputs_list[0].shape[1] * inputs_list[0].shape[2]), dtype=np.float32),
            np.empty((0, leaf_list[0].shape[1] * leaf_list[0].shape[2]), dtype=np.float32),
            np.empty((0, joints_list[0].shape[1] * joints_list[0].shape[2]), dtype=np.float32),
        )

    return (
        np.concatenate(all_inputs, axis=0),
        np.concatenate(all_leaf, axis=0),
        np.concatenate(all_joints, axis=0),
    )


def build_inputs(acc_list, quat_list):
    return [np.concatenate([acc, quat], axis=-1) for acc, quat in zip(acc_list, quat_list)]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(RANDOM_SEED)

    base_dir = Path(__file__).resolve().parent
    processed_dir = base_dir / "processed_KIT"
    models_dir = base_dir / "models_KIT"
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
    n_val = max(1, int(n_files * VAL_RATIO))
    n_test = max(1, n_files - n_train - n_val)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # 确保至少有一个测试文件
    if not test_files:
        test_files = val_files[-1:]
        val_files = val_files[:-1]

    print(f"train文件数: {len(train_files)}")
    print(f"val文件数: {len(val_files)}")
    print(f"test文件数: {len(test_files)}")

    acc_tr, quat_tr, joints_tr, leaf_tr = load_split(train_files)
    acc_val, quat_val, joints_val, leaf_val = load_split(val_files)
    acc_te, quat_te, joints_te, leaf_te = load_split(test_files)

    acc_mean, acc_std = compute_norm_stats(acc_tr)
    quat_mean, quat_std = compute_norm_stats(quat_tr)
    leaf_mean, leaf_std = compute_norm_stats(leaf_tr)
    joints_mean, joints_std = compute_norm_stats(joints_tr)

    acc_tr = normalize_list(acc_tr, acc_mean, acc_std)
    acc_val = normalize_list(acc_val, acc_mean, acc_std)
    acc_te = normalize_list(acc_te, acc_mean, acc_std)

    quat_tr = normalize_list(quat_tr, quat_mean, quat_std)
    quat_val = normalize_list(quat_val, quat_mean, quat_std)
    quat_te = normalize_list(quat_te, quat_mean, quat_std)

    leaf_tr = normalize_list(leaf_tr, leaf_mean, leaf_std)
    leaf_val = normalize_list(leaf_val, leaf_mean, leaf_std)
    leaf_te = normalize_list(leaf_te, leaf_mean, leaf_std)

    joints_tr = normalize_list(joints_tr, joints_mean, joints_std)
    joints_val = normalize_list(joints_val, joints_mean, joints_std)
    joints_te = normalize_list(joints_te, joints_mean, joints_std)

    inputs_tr = build_inputs(acc_tr, quat_tr)
    inputs_val = build_inputs(acc_val, quat_val)
    inputs_te = build_inputs(acc_te, quat_te)

    X_train, leaf_train, joints_train = create_windows_from_lists(
        inputs_tr, leaf_tr, joints_tr, WINDOW_SIZE, STEP_SIZE_TRAIN
    )
    X_val, leaf_val_target, joints_val_target = create_windows_from_lists(
        inputs_val, leaf_val, joints_val, WINDOW_SIZE, STEP_SIZE_TRAIN
    )

    X_test_list, leaf_test_list, joints_test_list = [], [], []
    for inputs, leaf, joints in zip(inputs_te, leaf_te, joints_te):
        # 使用 STEP_SIZE_EVAL=1 构建滑动窗口，确保推理阶段能得到连续的逐帧输出
        X_w, leaf_w, joints_w = create_windows(inputs, leaf, joints, WINDOW_SIZE, STEP_SIZE_EVAL)
        X_test_list.append(X_w)
        leaf_test_list.append(leaf_w)
        joints_test_list.append(joints_w)

    if X_train.size == 0:
        raise RuntimeError("No training windows were generated. Check window_size and STEP_SIZE_TRAIN.")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(leaf_train),
        torch.from_numpy(joints_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    val_loader = None
    if X_val.size > 0:
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(leaf_val_target),
            torch.from_numpy(joints_val_target),
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[-1]
    leaf_size = leaf_train.shape[-1]
    joint_size = joints_train.shape[-1]

    model = PoseLSTM(
        input_size=input_size,
        leaf_output_size=leaf_size,
        full_output_size=joint_size,
    ).to(device)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = None
    best_epoch = None
    best_model_path = models_dir / "best_model.pth"

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_leaf_loss = 0.0
        train_full_loss = 0.0

        for X_batch, leaf_batch, joint_batch in train_loader:
            X_batch = X_batch.to(device)
            leaf_batch = leaf_batch.to(device)
            joint_batch = joint_batch.to(device)

            optimizer.zero_grad()
            leaf_pred, full_pred = model(X_batch)
            leaf_loss = criterion(leaf_pred, leaf_batch)
            full_loss = criterion(full_pred, joint_batch)
            loss = leaf_loss + full_loss
            loss.backward()
            optimizer.step()

            train_leaf_loss += leaf_loss.item()
            train_full_loss += full_loss.item()

        train_leaf_loss /= len(train_loader)
        train_full_loss /= len(train_loader)
        train_total_loss = train_leaf_loss + train_full_loss

        val_leaf_loss = None
        val_full_loss = None
        val_total_loss = None

        if val_loader is not None:
            model.eval()
            val_leaf_loss = 0.0
            val_full_loss = 0.0
            with torch.no_grad():
                for X_batch, leaf_batch, joint_batch in val_loader:
                    X_batch = X_batch.to(device)
                    leaf_batch = leaf_batch.to(device)
                    joint_batch = joint_batch.to(device)
                    leaf_pred, full_pred = model(X_batch)
                    leaf_loss = criterion(leaf_pred, leaf_batch)
                    full_loss = criterion(full_pred, joint_batch)
                    val_leaf_loss += leaf_loss.item()
                    val_full_loss += full_loss.item()

            val_leaf_loss /= len(val_loader)
            val_full_loss /= len(val_loader)
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
        )
        if val_total_loss is not None:
            msg += (
                f"ValLeaf: {val_leaf_loss:.4f}  "
                f"ValFull: {val_full_loss:.4f}  "
                f"ValTotal: {val_total_loss:.4f}  "
            )
        msg += f"Time: {epoch_time:.2f}s"
        print(msg)

    last_model_path = models_dir / "last_model.pth"
    torch.save(model.state_dict(), last_model_path)

    norm_params_path = models_dir / "norm_params.npz"
    np.savez(
        norm_params_path,
        acc_mean=acc_mean,
        acc_std=acc_std,
        quat_mean=quat_mean,
        quat_std=quat_std,
        leaf_mean=leaf_mean,
        leaf_std=leaf_std,
        joints_mean=joints_mean,
        joints_std=joints_std,
    )

    test_sequences = []
    for path, X_w, leaf_w, joints_w in zip(test_files, X_test_list, leaf_test_list, joints_test_list):
        if X_w.size == 0:
            continue
        test_sequences.append({
            "file": str(path),
            "inputs": torch.from_numpy(X_w),
            "leaf": torch.from_numpy(leaf_w),
            "joints": torch.from_numpy(joints_w),
        })

    train_sequences = []
    for path, inputs, leaf, joints in zip(train_files, inputs_tr, leaf_tr, joints_tr):
        X_w, leaf_w, joints_w = create_windows(inputs, leaf, joints, WINDOW_SIZE, STEP_SIZE_EVAL)
        if X_w.size == 0:
            continue
        train_sequences.append({
            "file": str(path),
            "inputs": torch.from_numpy(X_w),
            "leaf": torch.from_numpy(leaf_w),
            "joints": torch.from_numpy(joints_w),
        })

    test_data_path = test_data_dir / "test_sequences.pt"
    torch.save({
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE_EVAL,
        "sequences": test_sequences,
    }, test_data_path)

    train_data_path = test_data_dir / "train_sequences.pt"
    torch.save({
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE_EVAL,
        "sequences": train_sequences,
    }, train_data_path)

    summary_path = models_dir / "training_summary.json"
    summary = {
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "test_files": [str(p) for p in test_files],
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "last_model_path": str(last_model_path),
        "best_model_path": str(best_model_path),
        "norm_params_path": str(norm_params_path),
        "test_data_path": str(test_data_path),
        "train_data_path": str(train_data_path),
        "hyperparameters": {
            "window_size": WINDOW_SIZE,
            "step_size_train": STEP_SIZE_TRAIN,
            "step_size_eval": STEP_SIZE_EVAL,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")
    print(f"Normalization parameters saved to: {norm_params_path}")
    print(f"Test sequences saved to: {test_data_path}")
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
