import torch
import os
import numpy as np
from model_def import PoseLSTM
import pickle


# ------------------- 函数定义 ------------------- #
def denormalize(x_norm):
    return x_norm * joints_std + joints_mean 
 

# ------------------- 读取数据和模型 ------------------- #
# 读取测试数据
test_data_path = os.path.join("..", "test_data", "test_data.pkl")
with open(test_data_path, "rb") as f:
    data = pickle.load(f)
X_test_win_list = data["X_test_win_list"]
Y_test_win_list = data["Y_test_win_list"]

# 读取归一化参数
norm_params_path = os.path.join("..", "models", "norm_params.npz")
norm = np.load(norm_params_path)
joints_mean, joints_std = norm["joints_mean"], norm["joints_std"]

# 读取模型
model = PoseLSTM()
model_path = os.path.join("..", "models", "best_model.pth")
weights = torch.load(model_path, map_location=torch.device("cpu"))   # 读取事先保存的权重
model.load_state_dict(weights)


# ----------------- 将模型应用于全部测试数据 --------------------#
true_joints_cm_list = []
pred_joints_cm_list = []

model.eval()
for test_idx, (X_test_win, Y_test_win) in enumerate(zip(X_test_win_list, Y_test_win_list)):
    print(f"{len(X_test_win_list)}個のテストデータのうち、 {test_idx + 1} 個目を推定中...")

    # 转换为 torch.Tensor
    X_test_tensor = torch.tensor(X_test_win, dtype=torch.float32)

    # 使用模型进行预测
    with torch.no_grad():
        pred = model(X_test_tensor).numpy()  # shape (N, 72)

    # reshape
    true_joints = Y_test_win.reshape(-1, 24, 3)
    pred_joints = pred.reshape(-1, 24, 3)

    # 还原归一化
    true_joints_denorm = denormalize(true_joints)
    pred_joints_denorm = denormalize(pred_joints)

    # 转换为厘米单位
    true_joints_cm = true_joints_denorm * 100
    pred_joints_cm = pred_joints_denorm * 100

    # 加入列表
    true_joints_cm_list.append(true_joints_cm)
    pred_joints_cm_list.append(pred_joints_cm)

print("すべてのテストデータを推定しました。")


# ------------------ 设置保存文件夹 ------------------ #
base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts 文件夹
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True)


# ------------------ 保存推理结果 ------------------ #
results_path = os.path.join(results_dir, "all_results.pkl")

with open(results_path, "wb") as f:
    pickle.dump({
        "true_joints_cm_list": true_joints_cm_list,
        "pred_joints_cm_list": pred_joints_cm_list
    }, f)

print(f"すべての推定結果を保存しました: {results_path}")





# ------------------ 误差评估 ------------------ #
all_errors = []  # 将每个样本和关节的误差汇总

for true_joints_cm, pred_joints_cm in zip(true_joints_cm_list, pred_joints_cm_list):
    # shape: (N, 24, 3)
    diff = pred_joints_cm - true_joints_cm  # 差值
    dist = np.linalg.norm(diff, axis=2)     # 各关节的三维距离误差（cm）
    all_errors.append(dist)         

# 合并全部测试结果
all_errors = np.concatenate(all_errors, axis=0)  # 形状: (总帧数, 24)

# 计算各项指标
mae_per_joint = np.mean(all_errors, axis=0)        # 每个关节的平均误差 (cm)
mae_total = np.mean(all_errors)                    # 平均误差的平均值
pck_10 = np.mean(all_errors < 10) * 100   # 10cm 以内的比例 (%)

print("\n--- 推定精度評価 ---")
print("関節ごとの平均誤差[cm]:", " ".join([f"{x:.2f}" for x in mae_per_joint]))
print(f"全体の平均誤差: {mae_total:.2f} cm")
print(f"誤差10cm以内の割合: {pck_10:.2f}%")