import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


# ---------------------- 結果の読み込み ---------------------#
with open("../results/all_results.pkl", "rb") as f:
    data = pickle.load(f)
true_joints_cm_list = data["true_joints_cm_list"]
pred_joints_cm_list = data["pred_joints_cm_list"]

#可視化する動作データを設定
idx = 8
true_joints_cm = true_joints_cm_list[idx]
pred_joints_cm = pred_joints_cm_list[idx]

print(f"{len(true_joints_cm_list)}個の動作データのうち、 {idx + 1} 個目を可視化する")
print(f"フレーム数: {len(true_joints_cm)}")


# ---------------------- SMPL構造の関節接続定義 ---------------------#
SMPL_CONNECTIONS = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11), 
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),    
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), 
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), 
]


# ---------------------- 1フレームを描く関数 ---------------------#
def plot_smpl_frame(ax, joints, color):
    for (i, j) in SMPL_CONNECTIONS:
        ax.plot(
            [joints[i, 0], joints[j, 0]],
            [joints[i, 1], joints[j, 1]],
            [joints[i, 2], joints[j, 2]],
            color=color, linewidth=2
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=15)

# ---------------------- アニメーションの作成・表示 ---------------------#
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 軸範囲を固定（データに合わせて調整）
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-80, 80)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 初期化
true_plot, = ax.plot([], [], [], color='blue', lw=2, label='True')
pred_plot, = ax.plot([], [], [], color='red', lw=2, label='Pred')
ax.legend()

def update(frame):
    ax.cla()  # クリア
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-80, 80)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame}")

    # 真の関節
    plot_smpl_frame(ax, true_joints_cm[frame], color='blue')
    # 予測関節
    plot_smpl_frame(ax, pred_joints_cm[frame], color='red')

    return ax,

ani = FuncAnimation(fig, update, frames=len(true_joints_cm), interval=50)
plt.show()


# ------------------ アニメーションを保存 ------------------#
base_dir = os.path.dirname(os.path.abspath(__file__))  # scriptsフォルダ
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True)

# ループの中で idx ごとにファイル名を変える
mp4_path = os.path.join(results_dir, f"pose{idx + 1:03d}_visualization.mp4")
ani.save(mp4_path, writer="ffmpeg", fps=20)
print(f"アニメーションを保存しました: {mp4_path}")