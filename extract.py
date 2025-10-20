import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))

import numpy as np
from fstar_watanabe import STAR
from scipy.spatial.transform import Rotation as R

# ------------------- 関数 ------------------- #
EPSILON = 1e-3

def dot_product(v1, v2):
    return np.dot(v1, v2)

def vector(p1, p2):
    return p2 - p1

def get_points_by_indices(skin_verts, idx1, idx2, idx3):
    return skin_verts[idx1], skin_verts[idx2], skin_verts[idx3]

def is_right_triangle(skin_verts, idx1, idx2, idx3):
    A, B, C = get_points_by_indices(skin_verts, idx1, idx2, idx3)
    AB = vector(A, B)
    AC = vector(A, C)
    BC = vector(B, C)
    if abs(dot_product(AB, AC)) <= EPSILON or \
       abs(dot_product(AB, BC)) <= EPSILON or \
       abs(dot_product(AC, BC)) <= EPSILON:
        return True
    return False

def compute_normal(A, B, C):
    AB = vector(A, B)
    AC = vector(A, C)
    return np.cross(AB, AC)

def compute_mean_position(A, B, C):
    return (A + B + C) / 3

def normalize(v):
    return v / np.linalg.norm(v)

def compute_orientation(x, y, z):
    rot_matrix = np.vstack([normalize(x), normalize(y), normalize(z)]).T
    rotation = R.from_matrix(rot_matrix)
    return rotation.as_quat()

def simulate_IMU(skin_verts, idx1, idx2, idx3):
    A, B, C = get_points_by_indices(skin_verts, idx1, idx2, idx3)
    mean_position = compute_mean_position(A, B, C)
    normal = compute_normal(A, B, C)
    x_axis = normalize(vector(A, B))
    z_axis = normalize(normal)
    y_axis = np.cross(z_axis, x_axis)
    orientation_quat = compute_orientation(x_axis, y_axis, z_axis)
    return mean_position, orientation_quat

# ------------------- ディレクトリ設定 ------------------- #
target_dir = os.path.join(os.path.dirname(__file__), "../CMU")
output_dir = os.path.join(os.path.dirname(__file__), "../processed")
os.makedirs(output_dir, exist_ok=True)

# ------------------- CMUファイル一覧取得 ------------------- #
npz_files = []
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".npz"):
            npz_files.append(os.path.join(root, file))

print(f"見つかったファイル数: {len(npz_files)}")

# ------------------- ファイルごとに処理 ------------------- #
for file_path in npz_files:
    print(f"処理中: {file_path}")
    mocap_data = dict(np.load(file_path))
    gender = mocap_data['gender']
    betas = mocap_data['betas'][:10]

    star_model = STAR(gender=gender, num_betas=10)

    map_dict = {
        "right_wrist": [5663, 5662, 5644],
        "left_wrist": [2209, 2202, 2207],
        "pelvis": [1783, 3159, 5248],
        "head": [3163, 444, 3770],
        "right_ankle": [4846, 4847, 4642],
        "left_ankle": [1373, 1375, 1075],
    }

    pos_history = {name: [] for name in map_dict}
    quat_history = {name: [] for name in map_dict}
    joint_pos_history = []

    for i in range(mocap_data['poses'].shape[0]):
        pose = mocap_data['poses'][i, :72]
        trans = mocap_data['trans'][i]
        star_model.forward(pose=pose, trans=trans, betas=betas)
        joint_pos_history.append(star_model.J_transformed.copy())
        points = star_model.v.copy()

        for name, (i1, i2, i3) in map_dict.items():
            pos, quat = simulate_IMU(points, i1, i2, i3)
            pos_history[name].append(pos)
            quat_history[name].append(quat)

    names = list(map_dict.keys())
    pos_arr = np.stack([pos_history[n] for n in names], axis=1)
    quat_arr = np.stack([quat_history[n] for n in names], axis=1)
    joint_arr = np.stack(joint_pos_history, axis=0)

    delta_T = 1.0 / mocap_data["mocap_framerate"]
    acc_diff2 = np.diff(pos_arr, n=2, axis=0) / (delta_T ** 2)
    acc_arr = np.pad(acc_diff2, pad_width=((1, 1), (0, 0), (0, 0)), mode='constant')

    # 保存
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_processed.npz")
    np.savez(out_path, quat=quat_arr, acc=acc_arr, joints=joint_arr)
    print(f"保存完了: {out_path}")
