import os
import random
import glob
import re
import argparse
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from utils_visualization import visualize_pcd, visualize_pcd_plotly, visualize_pcd_multiple

def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def collate(batch):
    # print('collate batch: ', len(batch), len(batch[0]))
    return batch

def ego_motion_compensation(points, time_indice, tsfm):
    """
    Input (torch.Tensor):
        points:         [N, 3]
        time_indice:    [N]
        tsfm:           [n_frames, 4, 4]
    """
    point_tsfm = tsfm[time_indice.astype(int)]
    R, t = point_tsfm[:,:3,:3], point_tsfm[:,:3,3:4]
    rec_points = np.einsum('bij,bjk -> bik', R, points[:,0:3,None]) + t
    return np.squeeze(rec_points, axis=-1)

def reconstruct_sequence(points, time_indice, inst_labels, tsfm, n_frames):
    """
    Reconstruct a sequence of point clouds, this only works for batch_size = 1
    Input:
        points:         [N,3]
        time_indice:    [N]
        inst_labels:    [N]
        tsfm:           [M, n_frames, 4, 4]
        n_frames:       integer
    """
    assert n_frames == tsfm.shape[1]
    indice = (inst_labels * n_frames + time_indice).astype(int)
    points_tsfm = tsfm.reshape(-1, 4, 4)[indice]
    R, t = points_tsfm[:,:3,:3], points_tsfm[:,:3,3:4]
    rec_points = np.einsum('bij,bjk -> bik', R, points[:,0:3,None]) + t
    return np.squeeze(rec_points, axis=-1)