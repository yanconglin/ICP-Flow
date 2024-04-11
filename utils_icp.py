import numpy as np
import torch
import pytorch3d.ops as pytorch3d_ops
import pytorch3d.transforms as pytorch3d_t
import open3d as o3d
import seaborn as sns
import time
import scipy
import warnings
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_helper import transform_points_batch
from utils_icp_pytorch3d import iterative_closest_point
from utils_helper import nearest_neighbor_batch, random_choice
warnings.filterwarnings('ignore')

def apply_icp(args, src, dst, init_poses):
    src_tmp = transform_points_batch(src, init_poses)

    Rts = pytorch3d_icp(args, src_tmp, dst)
    Rts = torch.bmm(Rts, init_poses)

    # # # pytorch 3d icp might go wrong ! to fix!
    mask_src = src[:, : , -1] > 0.0
    _, error_init = nearest_neighbor_batch(src_tmp, dst)
    error_init = (error_init * mask_src).sum(dim=1) / mask_src.sum(dim=1)

    src_tmp = transform_points_batch(src, Rts)
    _, error_icp = nearest_neighbor_batch(src_tmp, dst)
    error_icp = (error_icp * mask_src).sum(dim=1) / mask_src.sum(dim=1)
    invalid =  error_icp>=error_init
    Rts[invalid] = init_poses[invalid]

    # # # # visualization
    # k = 0
    # print('init_pose: ', init_poses[k], Rts[k])
    # print('rot invalid, roll back to init pose, Rt: ', invalid[k], error_init[k], error_icp[k])
    # src_mask = src[:, :, 3]>0
    # dst_mask = dst[:, :, 3]>0
    # src_tmp = transform_points_batch(src, Rts)
    # visualize_pcd(np.concatenate([src_tmp[k, src_mask[k]].cpu().numpy(), dst[k, dst_mask[k]].cpu().numpy()], axis=0), 
    #               np.concatenate([np.zeros((sum(src_mask[k])))+1, np.zeros((sum(dst_mask[k])))+2], axis=0), 
    #               num_colors=3,
    #               title=f'after icp, size: {sum(src_mask[k])} vs {sum(dst_mask[k])}')
    return Rts

def pytorch3d_icp(args, src, dst):
    icp_result = iterative_closest_point(src, dst, 
                                            init_transform=None, 
                                            thres=args.thres_dist,
                                            max_iterations=100,
                                            relative_rmse_thr=1e-6,
                                            estimate_scale=False,
                                            allow_reflection=False,
                                            verbose=False)

    Rs = icp_result.RTs.R
    ts = icp_result.RTs.T

    Rts = torch.cat([Rs, ts[:, None, :]], dim=1) 
    Rts = torch.cat([Rts.permute(0, 2, 1), Rts.new_zeros(len(ts), 1, 4)], dim=1)
    Rts[:, 3, 3]=1.0

    # print('pytorch3d icp Rt: ', Rt)
    # src_tmp = transform_points_tensor(src, Rt)
    # visualize_pcd(np.concatenate([src_tmp.cpu().numpy(), dst.cpu().numpy()], axis=0), 
    #               np.concatenate([np.zeros((len(src)))+1, np.zeros((len(dst)))+2], axis=0), 
    #               num_colors=3,
    #               title=f'registration debug, size: {len(src)} vs {len(dst)}')
    return Rts

