import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model_def import PoseLSTM, PoseTransformer, PoseTransformerCond, PoseMLP


# ------------------- 运行配置（直接修改下方变量） ------------------- #
# 日本語メモ:
# - まず触るなら EPOCHS / BATCH_SIZE / LR。
# - MODEL_TYPE でモデル構成を切り替えられる。
DATA_DIR = "data"        # 目录下所有 npz 即数据集
OUTPUT_DIR = Path("lstm")       # 模型与归一化统计保存目录
MODEL_TYPE = "lstm"            # 可选: "lstm"、"transformer"、"transformer_cond"
EPOCHS = 100
BATCH_SIZE = 100
LR = 2e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
SEED = 42
# DataLoader 相关配置
# 日本語メモ:
# - NUM_WORKERS はデータ読み込みの並列数。0だと単一プロセス。
NUM_WORKERS = 0
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
# 分段训练长度
# 日本語メモ:
# - 長い系列を一定長で分割して学習するための長さ。
SEGMENT_LEN = 100
# 早停策略
# 日本語メモ:
# - 検証損失が改善しない状態が続いたら打ち切る。
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.0
# 骨骼长度损失权重
# 日本語メモ:
# - 0なら骨長の正則化を使わない。
BONE_LOSS_WEIGHT = 1

# SMPL 连接关系
# 日本語メモ:
# - (parent, child) の骨リンク一覧。骨長損失で使う。
EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]
# Transformer 相关超参（仅在 MODEL_TYPE="transformer" 时使用）
# 日本語メモ:
# - Transformerの層数や隠れ次元の設定。
TF_D_MODEL = 256
TF_NHEAD = 8
TF_LAYERS = 4
TF_FF = 512
TF_DROPOUT = 0.1
WANDB_PROJECT = "Pose_Train"
WANDB_RUN_NAME = "train_seq"


# ------------------- 数据集 ------------------- #
# 日本語メモ:
# - npz には "x" と "y" があり、基本は (T, 72) 形状。
class SequenceDataset(Dataset):
    """按需从 npz 路径加载完整序列 (x, y)。"""

    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        data = np.load(path, mmap_mode="r")
        x = data["x"]
        y = data["y"]
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def collate_fn(batch):
    """按最大长度填充，对齐批次。"""
    # 日本語メモ:
    # - 可変長系列なので padding して長さをそろえる。
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True)
    return x_pad, y_pad, lengths


# ------------------- 工具函数 ------------------- #
# 日本語メモ:
# - 末端(leaf)関節だけを抜き出して軽い損失に使うことがある。
LEAF_IDS = [0, 7, 8, 12, 20, 21]  # 叶节点在 24 关节索引


def split_leaf_full(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    接受 y 为 (B, T, 72) 或 (B, T, 24, 3)，输出：
      叶节点 (B, T, 18), 全关节 (B, T, 72)
    """
    # 日本語メモ:
    # - 72次元 = 24関節 × (x,y,z)。
    if y.dim() == 3 and y.shape[-1] == 72:
        b, t, _ = y.shape
        y_view = y.view(b, t, 24, 3)
    elif y.dim() == 4 and y.shape[-2:] == (24, 3):
        b, t = y.shape[:2]
        y_view = y
        y = y.reshape(b, t, -1)
    else:
        raise ValueError(f"Unexpected y shape: {y.shape}")

    leaf = y_view[:, :, LEAF_IDS, :].reshape(b, t, -1)
    full = y.reshape(b, t, -1)
    return leaf, full


def to_joints_tensor(y: torch.Tensor) -> torch.Tensor:
    # 日本語メモ:
    # - 72次元を (24,3) に並べ替えるだけ。
    if y.dim() == 3 and y.shape[-1] == 72:
        b, t, _ = y.shape
        return y.view(b, t, 24, 3)
    if y.dim() == 4 and y.shape[-2:] == (24, 3):
        return y
    raise ValueError(f"Unexpected y shape: {y.shape}")


def bone_length_loss(
    full_pred: torch.Tensor,
    target_lengths: torch.Tensor,
    mask: torch.Tensor,
    parent_idx: torch.Tensor,
    child_idx: torch.Tensor,
) -> torch.Tensor:
    # 日本語メモ:
    # - 骨の長さが一定になるように正則化する損失。
    joints = to_joints_tensor(full_pred)
    diff = joints[:, :, child_idx, :] - joints[:, :, parent_idx, :]
    lengths = torch.norm(diff, dim=-1)
    target = target_lengths.view(1, 1, -1)
    mask_exp = mask.unsqueeze(-1)
    loss = ((lengths - target) ** 2) * mask_exp
    denom = mask_exp.sum().clamp(min=1.0) * lengths.shape[-1]
    return loss.sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, T, D); mask: (B, T) 为1表示有效帧。
    """
    # 日本語メモ:
    # - padding 部分は学習に使わないので mask で無視する。
    mask_exp = mask.unsqueeze(-1)
    diff = (pred - target) ** 2 * mask_exp
    denom = mask_exp.sum().clamp(min=1.0)
    return diff.sum() / denom


def set_seed(seed: int):
    # 日本語メモ:
    # - 乱数を固定して再現性を高める。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_bone_mean(paths: List[Path]) -> np.ndarray:
    # 日本語メモ:
    # - 訓練データから平均骨長を作り、骨長損失の基準にする。
    parents = np.array([p for p, _ in EDGES], dtype=np.int64)
    children = np.array([c for _, c in EDGES], dtype=np.int64)
    sum_lengths = np.zeros(len(EDGES), dtype=np.float64)
    total_frames = 0

    for p in paths[:100]:
        data = np.load(p, mmap_mode="r")
        y = data["y"]
        if y.ndim == 2 and y.shape[1] == 72:
            joints = y.reshape(-1, 24, 3)
        elif y.ndim == 3 and y.shape[1:] == (24, 3):
            joints = y
        else:
            raise ValueError(f"Unexpected y shape in {p}: {y.shape}")

        diff = joints[:, children, :] - joints[:, parents, :]
        lengths = np.linalg.norm(diff, axis=2)
        sum_lengths += lengths.sum(axis=0)
        total_frames += lengths.shape[0]

    if total_frames == 0:
        raise RuntimeError("No frames found when computing bone lengths.")
    return (sum_lengths / total_frames).astype(np.float32)

# ------------------- 训练主流程 ------------------- #
# 日本語メモ:
# - 学習ループ本体。訓練/検証を回してモデルを保存する。
def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
):
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model_type": MODEL_TYPE,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_ratio": val_ratio,
            "segment_len": SEGMENT_LEN,
            "bone_loss_weight": BONE_LOSS_WEIGHT,
            "tf_d_model": TF_D_MODEL,
            "tf_nhead": TF_NHEAD,
            "tf_layers": TF_LAYERS,
            "tf_ff": TF_FF,
            "tf_dropout": TF_DROPOUT,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
        },
    )

    all_paths = sorted(Path(data_dir).glob("*.npz"))
    if not all_paths:
        raise RuntimeError(f"目录中未找到 npz 数据: {data_dir}")

    # 划分训练/验证
    # 日本語メモ:
    # - データを訓練と検証に分割する。
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_paths))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1 - val_ratio)))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[-1:]

    train_paths = [all_paths[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]

    print("Computing mean bone lengths...")
    bone_mean = compute_bone_mean(train_paths)

    train_loader = DataLoader(
        SequenceDataset(train_paths),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        SequenceDataset(val_paths),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 日本語メモ:
    # - GPUがあればcuda、それ以外はCPU。
    bone_target = torch.from_numpy(bone_mean).to(device)
    parent_idx = torch.tensor([p for p, _ in EDGES], device=device, dtype=torch.long)
    child_idx = torch.tensor([c for _, c in EDGES], device=device, dtype=torch.long)
    if MODEL_TYPE == "lstm":
        # 日本語メモ:
        # - LSTMは時系列に強いベーシックモデル。
        model = PoseLSTM(
            input_size=72,
            leaf_output_size=len(LEAF_IDS) * 3,
            full_output_size=24 * 3,
        ).to(device)
    elif MODEL_TYPE == "transformer":
        # 日本語メモ:
        # - Transformerは自己注意で長距離依存を扱える。
        model = PoseTransformer(
            input_size=72,
            full_output_size=24 * 3,
            d_model=TF_D_MODEL,
            nhead=TF_NHEAD,
            num_layers=TF_LAYERS,
            dim_feedforward=TF_FF,
            dropout=TF_DROPOUT,
        ).to(device)
    elif MODEL_TYPE == "transformer_cond":
        # 日本語メモ:
        # - 条件付きTransformer（leaf出力あり）。
        model = PoseTransformerCond(
            input_size=72,
            leaf_output_size=len(LEAF_IDS) * 3,
            full_output_size=24 * 3,
            d_model=TF_D_MODEL,
            nhead=TF_NHEAD,
            num_layers=TF_LAYERS,

        ).to(device)
    elif MODEL_TYPE == "MLP":
        # 日本語メモ:
        # - MLPは時系列性を持たない基準モデル。
        model = PoseMLP(
            input_size=72,
            leaf_output_size=len(LEAF_IDS) * 3,
            full_output_size=24 * 3,
        ).to(device)    
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 日本語メモ:
    # - AdamWは一般的な学習アルゴリズム。
    best_val = None
    best_epoch = None
    no_improve = 0
    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"

    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_leaf_loss = 0.0
        train_full_loss = 0.0
        train_bone_loss = 0.0
        batches = 0

        for x_batch, y_batch, lengths in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # 日本語メモ:
            # - maskは有効フレーム=1、padding=0。
            mask = (torch.arange(x_batch.shape[1], device=device)[None, :] < lengths.to(device)[:, None]).float()

            segment_indices = []
            max_len = x_batch.shape[1]
            for start in range(0, max_len, SEGMENT_LEN):
                end = min(start + SEGMENT_LEN, max_len)
                mask_seg = mask[:, start:end]
                if mask_seg.sum().item() == 0:
                    continue
                segment_indices.append((start, end))

            if not segment_indices:
                continue

            optimizer.zero_grad()
            batch_leaf = 0.0
            batch_full = 0.0
            batch_bone = 0.0
            for start, end in segment_indices:
                x_seg = x_batch[:, start:end]
                y_seg = y_batch[:, start:end]
                mask_seg = mask[:, start:end]
                # 日本語メモ:
                # - Transformer用の padding マスク。
                key_padding_mask = (mask_seg == 0).bool()

                if MODEL_TYPE == "lstm":
                    leaf_pred, full_pred = model(x_seg)
                    leaf_gt, full_gt = split_leaf_full(y_seg)
                    leaf_loss = masked_mse(leaf_pred, leaf_gt, mask_seg)
                    full_mse = masked_mse(full_pred, full_gt, mask_seg)
                elif MODEL_TYPE == "transformer":
                    full_pred = model(x_seg, key_padding_mask=key_padding_mask)
                    _, full_gt = split_leaf_full(y_seg)
                    leaf_loss = torch.tensor(0.0, device=device)
                    full_mse = masked_mse(full_pred, full_gt, mask_seg)
                elif MODEL_TYPE == "MLP":
                    leaf_pred, full_pred = model(x_seg)
                    leaf_gt, full_gt = split_leaf_full(y_seg)
                    leaf_loss = masked_mse(leaf_pred, leaf_gt, mask_seg)
                    full_mse = masked_mse(full_pred, full_gt, mask_seg)
                else:  # transformer_cond
                    leaf_pred, full_pred = model(x_seg, key_padding_mask=key_padding_mask)
                    leaf_gt, full_gt = split_leaf_full(y_seg)
                    leaf_loss = masked_mse(leaf_pred, leaf_gt, mask_seg)
                    full_mse = masked_mse(full_pred, full_gt, mask_seg)

                bone_loss = bone_length_loss(
                    full_pred,
                    bone_target,
                    mask_seg,
                    parent_idx,
                    child_idx,
                )
                # 日本語メモ:
                # - full損失 + 骨長損失（重み付き）。
                full_loss = full_mse + BONE_LOSS_WEIGHT * bone_loss
                loss = (leaf_loss + full_loss) / len(segment_indices)
                loss.backward()
                batch_leaf += leaf_loss.item()
                batch_full += full_loss.item()
                batch_bone += bone_loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_leaf_loss += batch_leaf / len(segment_indices)
            train_full_loss += batch_full / len(segment_indices)
            train_bone_loss += batch_bone / len(segment_indices)
            batches += 1

        train_leaf_loss /= max(1, batches)
        train_full_loss /= max(1, batches)
        train_total = train_leaf_loss + train_full_loss

        # 验证
        # 日本語メモ:
        # - 検証は勾配を計算しない（no_grad）。
        model.eval()
        val_leaf_loss = 0.0
        val_full_loss = 0.0
        val_bone_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch, lengths in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask = (torch.arange(x_batch.shape[1], device=device)[None, :] < lengths.to(device)[:, None]).float()

                segment_indices = []
                max_len = x_batch.shape[1]
                for start in range(0, max_len, SEGMENT_LEN):
                    end = min(start + SEGMENT_LEN, max_len)
                    mask_seg = mask[:, start:end]
                    if mask_seg.sum().item() == 0:
                        continue
                    segment_indices.append((start, end))

                if not segment_indices:
                    continue

                batch_leaf = 0.0
                batch_full = 0.0
                batch_bone = 0.0
                for start, end in segment_indices:
                    x_seg = x_batch[:, start:end]
                    y_seg = y_batch[:, start:end]
                    mask_seg = mask[:, start:end]
                    key_padding_mask = (mask_seg == 0).bool()

                    if MODEL_TYPE == "lstm":
                        leaf_pred, full_pred = model(x_seg)
                        leaf_gt, full_gt = split_leaf_full(y_seg)
                        leaf_loss = masked_mse(leaf_pred, leaf_gt, mask_seg)
                        full_mse = masked_mse(full_pred, full_gt, mask_seg)
                    elif MODEL_TYPE == "transformer":
                        full_pred = model(x_seg, key_padding_mask=key_padding_mask)
                        _, full_gt = split_leaf_full(y_seg)
                        leaf_loss = torch.tensor(0.0, device=device)
                        full_mse = masked_mse(full_pred, full_gt, mask_seg)
                    elif MODEL_TYPE == "MLP":
                        leaf_pred, full_pred = model(x_seg)
                        leaf_gt, full_gt = split_leaf_full(y_seg)
                        leaf_loss = masked_mse(leaf_pred, leaf_gt, mask_seg)
                        full_mse = masked_mse(full_pred, full_gt, mask_seg)
                    else:
                        leaf_pred, full_pred = model(x_seg, key_padding_mask=key_padding_mask)
                        leaf_gt, full_gt = split_leaf_full(y_seg)
                        leaf_loss = masked_mse(leaf_pred, leaf_gt, mask_seg)
                        full_mse = masked_mse(full_pred, full_gt, mask_seg)

                    bone_loss = bone_length_loss(
                        full_pred,
                        bone_target,
                        mask_seg,
                        parent_idx,
                        child_idx,
                    )
                    full_loss = full_mse + BONE_LOSS_WEIGHT * bone_loss

                    batch_leaf += leaf_loss.item()
                    batch_full += full_loss.item()
                    batch_bone += bone_loss.item()

                val_leaf_loss += batch_leaf / len(segment_indices)
                val_full_loss += batch_full / len(segment_indices)
                val_bone_loss += batch_bone / len(segment_indices)
                val_batches += 1

        val_leaf_loss /= max(1, val_batches)
        val_full_loss /= max(1, val_batches)
        val_total = val_leaf_loss + val_full_loss

        epoch_time = time.perf_counter() - epoch_start
        wandb.log({
            "epoch": epoch,
            "train_leaf_loss": train_leaf_loss,
            "train_full_loss": train_full_loss,
            "train_bone_loss": train_bone_loss,
            "train_total_loss": train_total,
            "val_leaf_loss": val_leaf_loss,
            "val_full_loss": val_full_loss,
            "val_bone_loss": val_bone_loss,
            "val_total_loss": val_total,
            "epoch_time_s": epoch_time,
        })
        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLeaf: {train_leaf_loss:.4f}  TrainFull: {train_full_loss:.4f}  "
            f"TrainBone: {train_bone_loss:.4f}  TrainTotal: {train_total:.4f} | "
            f"ValLeaf: {val_leaf_loss:.4f}  ValFull: {val_full_loss:.4f}  "
            f"ValBone: {val_bone_loss:.4f}  ValTotal: {val_total:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if best_val is None or val_total < best_val - EARLY_STOP_MIN_DELTA:
            # 日本語メモ:
            # - 検証損失が更新されたらbest_modelを保存。
            best_val = val_total
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch}).")
                break

    torch.save(model.state_dict(), last_path)
    total_time = time.perf_counter() - total_start

    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Total time: {total_time:.2f}s")
    wandb.finish()


if __name__ == "__main__":
    train(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
