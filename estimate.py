import torch
import os
import numpy as np
from model_def import PoseLSTM
import pickle


# ------------------- 関数定義 ------------------- #
def denormalize(x_norm):
    return x_norm * joints_std + joints_mean 
 

# ------------------- データとモデルの読み込み ------------------- #
#テストデータの読み込み
test_data_path = os.path.join("..", "test_data", "test_data.pkl")
with open(test_data_path, "rb") as f:
    data = pickle.load(f)
X_test_win_list = data["X_test_win_list"]
Y_test_win_list = data["Y_test_win_list"]

#正規化パラメータの読み込み
norm_params_path = os.path.join("..", "models", "norm_params.npz")
norm = np.load(norm_params_path)
joints_mean, joints_std = norm["joints_mean"], norm["joints_std"]

#モデルの読み込み
model = PoseLSTM()
model_path = os.path.join("..", "models", "best_model.pth")
weights = torch.load(model_path, map_location=torch.device("cpu"))   # 保存しておいた重みを読み込む
model.load_state_dict(weights)


# ----------------- 全テストデータにモデルを適用 --------------------#
true_joints_cm_list = []
pred_joints_cm_list = []

model.eval()
for test_idx, (X_test_win, Y_test_win) in enumerate(zip(X_test_win_list, Y_test_win_list)):
    print(f"{len(X_test_win_list)}個のテストデータのうち、 {test_idx + 1} 個目を推定中...")

    # torch.Tensorに変換
    X_test_tensor = torch.tensor(X_test_win, dtype=torch.float32)

    # モデルで予測
    with torch.no_grad():
        pred = model(X_test_tensor).numpy()  # shape (N, 72)

    # reshape
    true_joints = Y_test_win.reshape(-1, 24, 3)
    pred_joints = pred.reshape(-1, 24, 3)

    # 正規化を戻す
    true_joints_denorm = denormalize(true_joints)
    pred_joints_denorm = denormalize(pred_joints)

    # cm単位に変換
    true_joints_cm = true_joints_denorm * 100
    pred_joints_cm = pred_joints_denorm * 100

    # リストに追加
    true_joints_cm_list.append(true_joints_cm)
    pred_joints_cm_list.append(pred_joints_cm)

print("すべてのテストデータを推定しました。")


# ------------------ 保存フォルダの設定 ------------------ #
base_dir = os.path.dirname(os.path.abspath(__file__))  # scriptsフォルダ
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True)


# ------------------ 推論結果を保存 ------------------ #
results_path = os.path.join(results_dir, "all_results.pkl")

with open(results_path, "wb") as f:
    pickle.dump({
        "true_joints_cm_list": true_joints_cm_list,
        "pred_joints_cm_list": pred_joints_cm_list
    }, f)

print(f"すべての推定結果を保存しました: {results_path}")





# ------------------ 誤差の評価 ------------------ #
all_errors = []  # 各サンプル・関節の誤差を全部まとめる

for true_joints_cm, pred_joints_cm in zip(true_joints_cm_list, pred_joints_cm_list):
    # shape: (N, 24, 3)
    diff = pred_joints_cm - true_joints_cm  # 差分
    dist = np.linalg.norm(diff, axis=2)     # 各関節の3次元距離誤差（cm）
    all_errors.append(dist)         

# 全テストを結合
all_errors = np.concatenate(all_errors, axis=0)  # shape: (合計フレーム数, 24)

# 各指標の計算
mae_per_joint = np.mean(all_errors, axis=0)        # 関節ごとの平均誤差 (cm)
mae_total = np.mean(all_errors)                    # 平均誤差の平均
pck_10 = np.mean(all_errors < 10) * 100   # 10cm以内の割合 (%)

print("\n--- 推定精度評価 ---")
print("関節ごとの平均誤差[cm]:", " ".join([f"{x:.2f}" for x in mae_per_joint]))
print(f"全体の平均誤差: {mae_total:.2f} cm")
print(f"誤差10cm以内の割合: {pck_10:.2f}%")