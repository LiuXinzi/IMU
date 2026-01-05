import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model_def import PoseLSTM

# ------------------- 运行配置（直接修改下方变量） ------------------- #
DATA_DIR = "data"        # 目录下所有 npz 即数据集
OUTPUT_DIR = Path("models_seq")       # 模型与归一化统计保存目录
EPOCHS = 80
BATCH_SIZE = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.1
SEED = 42


# ------------------- 数据集 ------------------- #
class SequenceDataset(Dataset):
    """持有完整序列 (x, y) 的数据集，x: (T,72)，y: (T,72)。"""

    def __init__(self, sequences: List[Tuple[np.ndarray, np.ndarray]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.sequences[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def collate_fn(batch):
    """按最大长度填充，对齐批次。"""
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True)
    y_pad = pad_sequence(ys, batch_first=True)
    return x_pad, y_pad, lengths


# ------------------- 工具函数 ------------------- #
LEAF_IDS = [0, 7, 8, 12, 20, 21]  # 叶节点在 24 关节索引


def split_leaf_full(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    接受 y 为 (B, T, 72) 或 (B, T, 24, 3)，输出：
      叶节点 (B, T, 18), 全关节 (B, T, 72)
    """
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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, T, D); mask: (B, T) 为1表示有效帧。
    """
    mask_exp = mask.unsqueeze(-1)
    diff = (pred - target) ** 2 * mask_exp
    denom = mask_exp.sum().clamp(min=1.0)
    return diff.sum() / denom


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequences(paths: List[Path]) -> List[Tuple[np.ndarray, np.ndarray]]:
    sequences = []
    for p in paths:
        data = np.load(p)
        x = data["x"].astype(np.float32)  # (T, 72)
        y = data["y"].astype(np.float32)  # (T, 72)  (24*3)

        sequences.append((x, y))
    return sequences


# ------------------- 训练主流程 ------------------- #
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

    all_paths = sorted(Path(data_dir).glob("*.npz"))
    if not all_paths:
        raise RuntimeError(f"目录中未找到 npz 数据: {data_dir}")

    sequences = load_sequences(all_paths)

    # 划分训练/验证
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sequences))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1 - val_ratio)))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[-1:]

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]

    train_loader = DataLoader(
        SequenceDataset(train_seqs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SequenceDataset(val_seqs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseLSTM(
        input_size=72,
        leaf_output_size=len(LEAF_IDS) * 3,
        full_output_size=24 * 3,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = None
    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_leaf_loss = 0.0
        train_full_loss = 0.0
        batches = 0

        for x_batch, y_batch, lengths in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask = (torch.arange(x_batch.shape[1], device=device)[None, :] < lengths.to(device)[:, None]).float()

            optimizer.zero_grad()
            leaf_pred, full_pred = model(x_batch)
            leaf_gt, full_gt = split_leaf_full(y_batch)

            leaf_loss = masked_mse(leaf_pred, leaf_gt, mask)
            full_loss = masked_mse(full_pred, full_gt, mask)
            loss = leaf_loss + full_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_leaf_loss += leaf_loss.item()
            train_full_loss += full_loss.item()
            batches += 1

        train_leaf_loss /= max(1, batches)
        train_full_loss /= max(1, batches)
        train_total = train_leaf_loss + train_full_loss

        # 验证
        model.eval()
        val_leaf_loss = 0.0
        val_full_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch, lengths in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask = (torch.arange(x_batch.shape[1], device=device)[None, :] < lengths.to(device)[:, None]).float()

                leaf_pred, full_pred = model(x_batch)
                leaf_gt, full_gt = split_leaf_full(y_batch)

                leaf_loss = masked_mse(leaf_pred, leaf_gt, mask)
                full_loss = masked_mse(full_pred, full_gt, mask)

                val_leaf_loss += leaf_loss.item()
                val_full_loss += full_loss.item()
                val_batches += 1

        val_leaf_loss /= max(1, val_batches)
        val_full_loss /= max(1, val_batches)
        val_total = val_leaf_loss + val_full_loss

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLeaf: {train_leaf_loss:.4f}  TrainFull: {train_full_loss:.4f}  TrainTotal: {train_total:.4f} | "
            f"ValLeaf: {val_leaf_loss:.4f}  ValFull: {val_full_loss:.4f}  ValTotal: {val_total:.4f}"
        )

        if best_val is None or val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)

    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")


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
