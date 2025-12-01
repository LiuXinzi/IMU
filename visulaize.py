import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
# try:
#     from fstar_watanabe import STAR  # Optional
# except ImportError:
#     STAR = None
from fstar_watanabe import STAR


# # ---------- SMPL edges ----------
def get_smpl_edges():
   
   star = STAR(gender="neutral", num_betas=10, skip_forward=True)
   kintree = star.kintree_table
#             print("use STAR")
   return [(int(kintree[0, i]), int(kintree[1, i])) for i in range(1, kintree.shape[1])]


    
# def get_smpl_edges() -> list:
#     if STAR is not None:
#         try:
#             star = STAR(gender="neutral", num_betas=10)
#             kintree = star.kintree_table
#             print("use STAR")
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


# ---------- 让 XYZ 比例一致 ----------
def set_axes_equal(ax, coords):
    coords = coords.reshape(-1, 3)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2
    r = max(r, 1e-6)

    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)


# ---------- 直接可视化，不保存GIF ----------
def show_motion(joints):
    edges = get_smpl_edges()
    T, J, _ = joints.shape

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("CMU Motion Sequence (No GIF Saved)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter = ax.scatter([], [], [], c="tab:blue", s=20)
    lines = [ax.plot([], [], [], c="tab:blue")[0] for _ in edges]

    set_axes_equal(ax, joints)

    def update(f):
        pts = joints[f]
        scatter._offsets3d = (pts[:,0], pts[:,1], pts[:,2])

        for line, (p, c) in zip(lines, edges):
            line.set_data([pts[p,0], pts[c,0]], [pts[p,1], pts[c,1]])
            line.set_3d_properties([pts[p,2], pts[c,2]])

        return []

    # *** 必须保存到变量 anim ***
    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=30, blit=False
    )

    plt.show()  # 显示动画



# ---------- 主函数：加载 npz 并展示 ----------
def main():

    npz_path = "D:/IMU/Watanabe/processed_CMU_1/32_01_poses_processed.npz"   # TODO: 改成你的 npz 文件
    data = np.load(npz_path)

    if "joints" in data:
        joints = data["joints"]
    elif "preds" in data:
        joints = data["preds"]
    elif "targets" in data:
        joints = data["targets"]
    else:
        raise ValueError("npz 必须包含 joints / preds / targets 之一")

    # reshape if flattened
    if joints.ndim == 2:
        J = joints.shape[1] // 3
        joints = joints.reshape(-1, J, 3)

    print("Loaded:", joints.shape)
    show_motion(joints)


if __name__ == "__main__":
    main()
