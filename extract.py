import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))

import numpy as np
from fstar_watanabe import STAR
from scipy.spatial.transform import Rotation as R

# ------------------- 函数 ------------------- #
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

def synthesize_acceleration(pos, fps, smooth_n=4):
    """
    Synthesize accelerations from position sequences using a smoothed finite difference.
    """
    if pos.shape[0] < 3:
        return np.zeros_like(pos)
    factor = float(fps) ** 2
    core = (pos[:-2] + pos[2:] - 2 * pos[1:-1]) * factor
    acc = np.concatenate([np.zeros_like(core[:1]), core, np.zeros_like(core[:1])], axis=0)
    if smooth_n > 0 and pos.shape[0] > smooth_n * 2:
        refined = (pos[:-2 * smooth_n] + pos[2 * smooth_n:] - 2 * pos[smooth_n:-smooth_n]) * factor / (smooth_n ** 2)
        acc[smooth_n:-smooth_n] = refined
    return acc

def to_pelvis_frame(pos_arr, quat_arr, acc_arr, joint_arr, leaf_arr, pelvis_idx, root_joint_idx=0):
    """
    Convert IMU signals and joint positions into the pelvis local frame.
    """
    pelvis_pos = pos_arr[:, pelvis_idx]
    pelvis_acc = acc_arr[:, pelvis_idx]
    pelvis_quat = quat_arr[:, pelvis_idx]

    rel_pos = np.empty_like(pos_arr)
    rel_acc = np.empty_like(acc_arr)
    rel_quat = np.empty_like(quat_arr)
    rel_joint = np.empty_like(joint_arr)
    rel_leaf = np.empty_like(leaf_arr)

    pelvis_rot_inv = R.from_quat(pelvis_quat).inv()

    for f in range(quat_arr.shape[0]):
        inv_rot = pelvis_rot_inv[f]
        rel_pos[f] = inv_rot.apply(pos_arr[f] - pelvis_pos[f])
        rel_acc[f] = inv_rot.apply(acc_arr[f] - pelvis_acc[f])
        root_pos = joint_arr[f, root_joint_idx]
        rel_joint[f] = inv_rot.apply(joint_arr[f] - root_pos)
        rel_leaf[f] = inv_rot.apply(leaf_arr[f] - root_pos)
        sensor_rot = R.from_quat(quat_arr[f])
        rel_quat[f] = (inv_rot * sensor_rot).as_quat()

    return rel_pos, rel_quat, rel_acc, rel_joint, rel_leaf

def simulate_IMU(skin_verts, idx1, idx2, idx3):
    A, B, C = get_points_by_indices(skin_verts, idx1, idx2, idx3)
    mean_position = compute_mean_position(A, B, C)
    normal = compute_normal(A, B, C)
    x_axis = normalize(vector(A, B))
    z_axis = normalize(normal)
    y_axis = np.cross(z_axis, x_axis)
    orientation_quat = compute_orientation(x_axis, y_axis, z_axis)
    return mean_position, orientation_quat

# ------------------- 目录设置 ------------------- #
def process_dataset(target_dir=None, output_dir=None):
    if target_dir is None:
        target_dir = os.path.join(os.path.dirname(__file__), "a")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "processed_KIT")

    os.makedirs(output_dir, exist_ok=True)

    npz_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))

    print(f"見つかったファイル数: {len(npz_files)}")

    for file_path in npz_files:

        print(f"処理中: {file_path}")
        mocap_data = dict(np.load(file_path))
        # import ipdb;ipdb.set_trace()
        if "mocap_framerate" not in mocap_data.keys():
            continue
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
        leaf_joint_map = {
            "right_wrist": 21,
            "left_wrist": 20,
            "pelvis": 0,
            "head": 12,
            "right_ankle": 8,
            "left_ankle": 7,
        }

        pos_history = {name: [] for name in map_dict}
        quat_history = {name: [] for name in map_dict}
        joint_pos_history = []

        mocap_fps = float(mocap_data["mocap_framerate"])
        
        frame_step = 2 if mocap_fps >= 99.0 else 1
        effective_fps = mocap_fps / frame_step
        frame_indices = range(0, mocap_data['poses'].shape[0], frame_step)

        for frame_idx in frame_indices:
            pose = mocap_data['poses'][frame_idx, :72]
            trans = mocap_data['trans'][frame_idx]
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

        acc_arr = synthesize_acceleration(pos_arr, effective_fps, smooth_n=4)

        leaf_pos_arr = np.stack([joint_arr[:, leaf_joint_map[name]] for name in names], axis=1)

        pelvis_idx = names.index("pelvis")
        pos_rel, quat_rel, acc_rel, joint_rel, leaf_rel = to_pelvis_frame(
            pos_arr, quat_arr, acc_arr, joint_arr, leaf_pos_arr, pelvis_idx
        )

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_processed.npz")
        np.savez(
            out_path,
            pos=pos_rel.astype(np.float32),
            quat=quat_rel.astype(np.float32),
            acc=acc_rel.astype(np.float32),
            joints=joint_rel.astype(np.float32),
            leaf_pos=leaf_rel.astype(np.float32),
            framerate=np.float32(effective_fps),
        )
        print(f"保存完了: {out_path}")


if __name__ == "__main__":
    process_dataset()
