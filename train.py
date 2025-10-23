import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_def import PoseLSTM  # 从 “model_def.py” 文件中导入模型
import pickle


# ------------------- 超参数 ------------------- #
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
window_size = 50
step_size_train = 25
step_size_test = 1
batch_size = 32
epochs = 10
learning_rate = 1e-3


# ------------------- 函数定义 ------------------- #
def load_split(file_list):
    acc_list, quat_list, joints_list = [], [], []
    for fpath in file_list:
        data = np.load(fpath)
        acc_list.append(data["acc"])
        quat_list.append(data["quat"])
        joints_list.append(data["joints"])
    return acc_list, quat_list, joints_list

def pelvis_relative(joints_list):
    new_list = []
    for joints in joints_list:
        pelvis = joints[:, 0:1, :]
        joints_rel = joints - pelvis
        new_list.append(joints_rel)
    return new_list

def normalize(x, mean, std):
    return (x - mean) / std

def create_window(X, Y, window_size, step_size):
    X_windows, Y_windows = [], []
    n_frames = X.shape[0]
    X_flat = X.reshape(n_frames, -1)
    Y_flat = Y.reshape(n_frames, -1)
    for start in range(0, n_frames - window_size + 1, step_size):
        end = start + window_size
        X_windows.append(X_flat[start:end])  # (window_size, feature_dim)
        Y_windows.append(Y_flat[end - 1])    # 窗口的最后一帧
    return np.array(X_windows), np.array(Y_windows)

def create_windows_from_lists(X_list, Y_list, window_size, step_size):
    X_windows_all, Y_windows_all = [], []
    for X, Y in zip(X_list, Y_list):
        X_w, Y_w = create_window(X, Y, window_size, step_size)
        X_windows_all.append(X_w)
        Y_windows_all.append(Y_w)
    # 合并所有窗口
    X_windows = np.concatenate(X_windows_all, axis=0)
    Y_windows = np.concatenate(Y_windows_all, axis=0)
    return X_windows, Y_windows


# ------------------- 数据读取与划分 ------------------- #
processed_dir = "../processed"
files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith(".npz")]

# 以文件为单位打乱顺序
np.random.seed(123)
np.random.shuffle(files)

# 仅选取 20 个文件（设置 np.random.seed(123) 时每次都会选中相同的文件）
files = files[:100]

# 按文件划分数据
n_files = len(files)
n_train = int(n_files * TRAIN_RATIO)
n_val = int(n_files * VAL_RATIO)
train_files = files[:n_train]
val_files = files[n_train:n_train+n_val]
test_files = files[n_train+n_val:]
print(f"trainファイル数: {len(train_files)}")
print(f"valファイル数: {len(val_files)}")
print(f"testファイル数: {len(test_files)}")

# 读取并合并各个集合
acc_list_train, quat_list_train, joints_list_train = load_split(train_files)
acc_list_val, quat_list_val, joints_list_val       = load_split(val_files)
acc_list_test, quat_list_test, joints_list_test    = load_split(test_files)

# 相对于骨盆基准进行坐标转换
joints_list_train = pelvis_relative(joints_list_train)
joints_list_val   = pelvis_relative(joints_list_val)
joints_list_test  = pelvis_relative(joints_list_test)


# ------------------- 归一化 ------------------- #
# 仅为了计算整个训练集的均值和方差而拼接
acc_all_train    = np.concatenate(acc_list_train, axis=0)
quat_all_train   = np.concatenate(quat_list_train, axis=0)
joints_all_train = np.concatenate(joints_list_train, axis=0)

acc_mean, acc_std       = acc_all_train.mean(axis=(0, 1)), acc_all_train.std(axis=(0, 1)) + 1e-6
quat_mean, quat_std     = quat_all_train.mean(axis=(0, 1)), quat_all_train.std(axis=(0, 1)) + 1e-6
joints_mean, joints_std = joints_all_train.mean(axis=(0, 1)), joints_all_train.std(axis=(0, 1)) + 1e-6

# 对每个文件做归一化（保持边界）
acc_list_train = [normalize(a, acc_mean, acc_std) for a in acc_list_train]
acc_list_val   = [normalize(a, acc_mean, acc_std) for a in acc_list_val]
acc_list_test  = [normalize(a, acc_mean, acc_std) for a in acc_list_test]

quat_list_train = [normalize(q, quat_mean, quat_std) for q in quat_list_train]
quat_list_val   = [normalize(q, quat_mean, quat_std) for q in quat_list_val]
quat_list_test  = [normalize(q, quat_mean, quat_std) for q in quat_list_test]

joints_list_train = [normalize(j, joints_mean, joints_std) for j in joints_list_train]
joints_list_val   = [normalize(j, joints_mean, joints_std) for j in joints_list_val]
joints_list_test  = [normalize(j, joints_mean, joints_std) for j in joints_list_test]


# ------------------- 构建 X 和 Y ------------------- #
# X : acc + quat
X_list_train = [np.concatenate([a, q], axis=-1) for a, q in zip(acc_list_train, quat_list_train)]
X_list_val   = [np.concatenate([a, q], axis=-1) for a, q in zip(acc_list_val, quat_list_val)]
X_list_test  = [np.concatenate([a, q], axis=-1) for a, q in zip(acc_list_test, quat_list_test)]
# Y : joints
Y_list_train = joints_list_train
Y_list_val   = joints_list_val
Y_list_test  = joints_list_test

# flatten((N, 6, 7) → (N, 42)) → 构建窗口
X_train_win, Y_train_win = create_windows_from_lists(X_list_train, Y_list_train, window_size, step_size_train)
X_val_win, Y_val_win     = create_windows_from_lists(X_list_val, Y_list_val, window_size, step_size_train)

# 测试用的 X、Y 想按动作可视化，因此保持列表并按窗口划分
X_test_win_list, Y_test_win_list = [], []
for X, Y in zip(X_list_test, Y_list_test):
    # 扁平化后生成窗口
    X_win, Y_win = create_window(X, Y, window_size, step_size_test)
    X_test_win_list.append(X_win)
    Y_test_win_list.append(Y_win)


# ------------------- DataLoader ------------------- #
X_train_tensor = torch.tensor(X_train_win, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_win, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_win, dtype=torch.float32)
Y_val_tensor   = torch.tensor(Y_val_win, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, Y_val_tensor), batch_size=batch_size, shuffle=False)


# ------------------- 训练 ------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)


    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")


# ------------------ 设置保存文件夹 ------------------ #
base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts 文件夹
models_dir = os.path.join(base_dir, "../models")
test_data_dir = os.path.join(base_dir, "../test_data")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)


# ------------------ 保存归一化参数 ----------------- #
norm_params_path = os.path.join(models_dir, "norm_params.npz")
np.savez(norm_params_path,
         joints_mean=joints_mean, joints_std=joints_std)
print(f"正規化パラメータを保存しました: {norm_params_path}")


# ------------------ 保存模型 ------------------ #
model_path = os.path.join(models_dir, "best_model.pth")
torch.save(model.state_dict(), model_path)
print(f"モデルを保存しました: {model_path}")


# ------------------ 保存测试数据 ------------------ #
test_data_path = os.path.join(test_data_dir, "test_data.pkl")
with open(test_data_path, "wb") as f:
    pickle.dump({
        "X_test_win_list": X_test_win_list,
        "Y_test_win_list": Y_test_win_list
    }, f)

print(f"テストデータを保存しました: {test_data_path}")