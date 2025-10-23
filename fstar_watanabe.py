# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V.
# (MPG). All rights reserved.
#
# 如果你在科研中使用此代码，请引用:
# STAR: Sparse Trained Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>

import numpy as np
import os
# import cv2
import time
from loguru import logger
import numba

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        val = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'"{func.__name__}" func took {end - start:.6f} seconds.')
        return val
    return wrapper

# 用于存储计算结果及附加属性，便于后续用 np.array(model) 获取数值数据
class ModelResult:
    def __init__(self, data):
        self.data = data
    def __array__(self, dtype=None):
        return np.array(self.data, dtype=dtype)

class STAR:
    def __init__(self, gender='male', num_betas=10):
        if gender not in ['male', 'female', 'neutral']:
            raise RuntimeError('Invalid model gender!')
        if num_betas < 2:
            raise RuntimeError('Number of betas should be at least 2')

        if gender == 'male':
            fname = os.path.join(os.path.dirname(__file__), 'star', 'male', 'model.npz')
        elif gender == 'female':
            fname = os.path.join(os.path.dirname(__file__), 'star', 'female', 'model.npz')
        else:
            fname = os.path.join(os.path.dirname(__file__), 'star', 'neutral', 'model.npz')
    
        if not os.path.exists(fname):
            raise RuntimeError('Path does not exist %s' % fname)
    
        model_dict = np.load(fname, allow_pickle=True)
        self.trans = np.zeros(3)
        self.posedirs = model_dict['posedirs']
        self.v_template = model_dict['v_template']
        self.J_regressor = model_dict['J_regressor']  # Regressor of the model
        self.weights = model_dict['weights']          # Weights
        self.kintree_table = model_dict['kintree_table']
        self.f = model_dict['f']
        self.shapedirs = model_dict['shapedirs'][:, :, :num_betas]  # Shape corrective blend shapes
        self.num_joints = self.weights.shape[1]
        self.num_betas = num_betas
        
        self.forward()
    
    
    # @timeit
    def forward(self, pose=None, trans=None, betas=None):
        if trans is None:
            trans = np.zeros(3)
        if pose is None:
            pose = np.zeros((self.num_joints * 3))
        if betas is None:
            betas = np.zeros(self.num_betas)
            
        self.pose = pose
        self.betas = betas
        self.trans = trans
        
        model = verts_decorated_quat(
            trans=self.trans,
            pose=self.pose,
            v_template=self.v_template,
            J_regressor=self.J_regressor,
            weights=self.weights,
            kintree_table=self.kintree_table,
            f=self.f,
            posedirs=self.posedirs,
            betas=self.betas,
            shapedirs=self.shapedirs,
            want_Jtr=True
        )
        
        self.v = np.array(model)
        self.J = model.J
        self.v_posed = model.v_posed
        self.v_shaped = model.v_shaped
        self.J_transformed = model.J_transformed
        self.model = model

def verts_decorated_quat(trans, pose, v_template, J_regressor, weights, kintree_table, f,
                           posedirs=None, betas=None, add_shape=True, shapedirs=None, want_Jtr=False):
    # 计算形状变化
    v_shaped = v_template + shapedirs @ betas # (V, 3)
    
    # 计算姿态相关的 blendshape
    quaternion_angles = axis2quat( pose.reshape((-1, 3))[1:] ).reshape(-1)  # 去掉root rot
    shape_feat = np.array([betas[1]])    # STAR 使用第2个 beta 值作为姿态补偿
    feat = np.concatenate([quaternion_angles, shape_feat], axis=0)
    poseblends = posedirs @ feat
    v_posed = v_shaped + poseblends

    # 计算关节位置（使用加权求和）
    J = J_regressor @ v_shaped

    result, meta = verts_core(pose, v_posed, J, weights, kintree_table, want_Jtr=True)
    tr = trans.reshape((1, 3))
    result = result + tr

    # 将结果封装到 ModelResult 对象中，便于附加额外属性
    model_result = ModelResult(result)
    model_result.trans = trans
    model_result.f = f
    model_result.pose = pose
    model_result.v_template = v_template
    model_result.J = J
    model_result.weights = weights
    model_result.posedirs = posedirs
    model_result.v_posed = v_posed
    model_result.shapedirs = shapedirs
    model_result.betas = betas
    model_result.v_shaped = v_shaped
    model_result.J_transformed = meta.Jtr if want_Jtr else None

    return model_result

def verts_core(pose, v, J, weights, kintree_table, want_Jtr=False):
    A, A_global = global_rigid_transformation(pose, J, kintree_table)
    # 计算 T = A.dot(weights.T)
    T = np.tensordot(A, weights.T, axes=([2], [0]))  # (4, 4, v)
    
    rest_shape_h = np.vstack((v.T, np.ones((1, v.shape[0]))))  # (4, v)
    v_new = (T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1)) +
              T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1)) +
              T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1)) +
              T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))).T
    v_new = v_new[:, :3]
    
    if want_Jtr:
        class ResultMeta:
            pass
        meta = ResultMeta()
        meta.Jtr = np.vstack([g[:3, 3] for g in A_global])
        meta.A = A
        meta.A_global = A_global
        meta.A_weighted = T
    else:
        meta = None
    
    return v_new, meta



@numba.njit(cache=True)
def global_rigid_transformation(pose, J, kintree_table):
    pose = pose.reshape((-1, 3))
    num_joints = kintree_table.shape[1]
    parent = dict(zip(kintree_table[1],kintree_table[0]))
    
    results_global = np.zeros((num_joints,4,4))
    results_global[:,3,3] = 1
    
    # 第一个关节
    R0 = rodrigues(pose[0, :])
    results_global[0,:3,:3] = R0
    results_global[0,:3, 3] = J[0, :]
    
    for i in range(1, num_joints):
        R = rodrigues(pose[i, :])
        diff = (J[i, :] - J[parent[i], :])
        results_global[i,:3,:3] = R
        results_global[i,:3, 3] = diff
        results_global[i] = results_global[parent[i]] @ results_global[i]
        
    results_stacked = results_global.copy()
    R_stacked = results_stacked[:,:3,:3]
    for i in range(num_joints):
        pack_val = R_stacked[i] @ J[i]
        results_stacked[i,:3,3] -= pack_val
    
    results_stacked = np.transpose( results_stacked, (1, 2, 0) )
    return results_stacked, results_global



@numba.njit(cache=True)
def axis2quat(p):
    # p: (N, 3)
    N = p.shape[0]
    q = np.empty((N, 4))
    for i in range(N):
        angle = np.sqrt(np.sum(p[i]**2))
        if angle < 1e-8:
            norm = np.array([0.0, 0.0, 0.0])
        else:
            norm = p[i] / angle
        cos_angle = np.cos(angle / 2.0)
        sin_angle = np.sin(angle / 2.0)
        q[i, 0] = norm[0] * sin_angle
        q[i, 1] = norm[1] * sin_angle
        q[i, 2] = norm[2] * sin_angle
        q[i, 3] = cos_angle - 1.0
    return q

@numba.njit(cache=True)
def rodrigues(r):
    theta = np.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
    R = np.empty((3, 3))
    if theta < 1e-8:
        # 返回单位矩阵
        for i in range(3):
            for j in range(3):
                R[i, j] = 0.0
            R[i, i] = 1.0
        return R
    k = r / theta
    # 构造反对称矩阵 K
    K = np.empty((3, 3))
    K[0, 0] = 0.0;    K[0, 1] = -k[2];  K[0, 2] = k[1]
    K[1, 0] = k[2];   K[1, 1] = 0.0;    K[1, 2] = -k[0]
    K[2, 0] = -k[1];  K[2, 1] = k[0];   K[2, 2] = 0.0
    # 构造单位矩阵 I
    I = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            I[i, j] = 0.0
        I[i, i] = 1.0
    # 计算 K^2 = K @ K
    K2 = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            tmp = 0.0
            for k_idx in range(3):
                tmp += K[i, k_idx] * K[k_idx, j]
            K2[i, j] = tmp
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    for i in range(3):
        for j in range(3):
            R[i, j] = I[i, j] + sin_theta * K[i, j] + (1 - cos_theta) * K2[i, j]
    return R

if __name__ == '__main__':
    s = STAR()
    s.forward(pose=np.random.rand(72),betas=np.random.rand(10))
    
    
    





