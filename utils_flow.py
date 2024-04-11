import os
import numpy as np
import open3d as o3d
import torch
import seaborn as sns
import scipy
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_helper import transform_points, transform_points_batch
from utils_eval import compute_epe_test
warnings.filterwarnings('ignore')
from scipy.spatial.transform import Rotation as R

def calculate_flow_rigid(points, transformation):
    points_tmp = transform_points(points, transformation)
    flow = points_tmp - points
    return flow

def flow_estimation(args,
                    src_points, dst_points,
                    src_labels, dst_labels, 
                    pairs, transformations, pose):
    unqs = np.unique(src_labels.astype(int))
    # sanity check:  src labels include -1e8 and -1; segment labels start from 0.
    # -1e8: ground, 
    # -1: unmatched/unclustered segments in non-ground points
    # assert len(unqs)-2 == max(src_label)+1
    # print('unqs: ', len(unqs), unqs)

    assert len(src_points) == len(src_labels)
    flow = np.zeros((len(src_points), 3)) 
    for unq in unqs:
        idxs = src_labels==unq
        xyz = src_points[src_labels==unq, 0:3]
        if unq in pairs[:, 0]:
            idx = np.flatnonzero(unq==pairs[:,0])
            assert len(idx)==1
            idx = idx.item()
            transformation = transformations[idx]
        else:
            transformation = np.eye(4)

        flow_i = calculate_flow_rigid(xyz, transformation @ pose)
        flow[idxs] = flow_i

    return flow

def calculate_flow_rigid(points, transformation):
    points_tmp = transform_points(points, transformation)
    flow = points_tmp - points
    return flow

def flow_estimation_torch(args,
                    src_points, dst_points,
                    src_labels, dst_labels, 
                    pairs, transformations, pose):
    assert len(src_points) == len(src_labels)
    T_per_point = torch.eye(4)[None, :, :].repeat(len(src_points), 1, 1)
    idxs_valid = src_labels[:, None] - pairs[:, 0][None, :]
    idxs_x,  idxs_y = torch.nonzero(idxs_valid == 0, as_tuple=True)
    T_per_point[idxs_x] = transformations[idxs_y]
    T_per_point = torch.bmm(T_per_point, pose[None, :, :].expand(len(src_points), 4, 4))
    src_points_h = torch.cat([src_points, src_points.new_ones(len(src_points), 1)], dim=-1)
    flow = torch.bmm(T_per_point, src_points_h[:, :, None])[:, 0:3, 0] - src_points
    return flow

# evaluate flow error for each segment, for debug only
def flow_evaluation(src_points, dst_points,
                    src_labels, dst_labels, 
                    flow_pd, flow_gt,
                    pose, transformations, 
                    pairs=None
                    ):
    unqs = np.unique(src_labels.astype(int))
    # sanity check:  src labels include -1e8 and -1; segment labels start from 0.
    # -1e8: ground, 
    # -1: unmatched/unclustered segments
    # assert len(unqs)-2 == max(src_label)+1
    flow = np.zeros((len(src_points), 3)) 
    # print(f'pairs: {pairs}')
    # per segment evaluation
    for unq in unqs:
        # print(f'calculate flow for segment: {k}')
        idx_i = src_labels==unq
        idx_j = dst_labels==unq
        xyz_i = src_points[idx_i, 0:3]
        xyz_j = dst_points[idx_j, 0:3]
        # calculate flow
        flow_pd_tmp = flow_pd[idx_i]
        flow_gt_tmp = flow_gt[idx_i]
        epe3d, acc3d_strict, acc3d_relax, outlier, Routlier = compute_epe_test(flow_pd_tmp, flow_gt_tmp)

        if unq<0: #  ground and non-clusterd points
        #     print(f'eval segment: {k:3d}, epe: {epe3d:.4f}, len_i: {len(xyz_i):6d}, len_j: {len(xyz_j):6d}, mean_i: {xyz_i.mean(0)}, mean_j: {xyz_j.mean(0)}')
        #     visualize_pcd_multiple(np.concatenate([xyz_i[:,0:3]+flow_pd_tmp, xyz_j[:,0:3]], axis=0), 
        #                            np.concatenate([xyz_i[:,0:3]+flow_pd_tmp, xyz_i[:,0:3]+flow_gt_tmp], axis=0), 
        #                            np.concatenate([np.zeros((len(xyz_i)))+1, np.zeros((len(xyz_j)))+2], axis=0), 
        #                            np.concatenate([np.zeros((len(xyz_i)))+1, np.zeros((len(xyz_i)))+2], axis=0),
        #                            num_colors=3,
        #                            title = f'eval segment {k}, flow error {epe3d}, i+flow vs i+flow_gt'
        #                            )
            continue
        idx = np.flatnonzero(unq==pairs[:, 0])
        if len(idx)==1: 
            idx = idx.item()
            print(f'eval segment: {unq:3d}, epe: {epe3d:.4f}, i: {int(pairs[idx,0]):3d}, j: {int(pairs[idx,1]):3d}; len_i: {len(xyz_i):6d}, len_j: {len(xyz_j):6d}, mean_i: {xyz_i.mean(0)}, mean_j: {xyz_j.mean(0)}')
        # print(f'                Running EPE: {epe3d:.4f}, ACC3DS: {acc3d_strict:.4f}, ACC3DR: {acc3d_relax:.4f}, Outlier: {outlier:.4f}')
        if epe3d>2.0: 
            # if k<0: continue
            label_i = np.zeros((len(src_labels)))-1
            label_j = np.zeros((len(dst_labels)))-1
            label_i[idx_i]=0
            label_j[idx_j]=1
            transform = transformations[idx]
            print('predictions with substantially large flow errors')
            print('matched pair: ', unq, pairs[idx], {len(xyz_i)}, {len(xyz_j)})
            print('pose: ', pose)
            print('transform: ', transform)
            print('translation: ', (xyz_i+flow_pd_tmp).mean(0), xyz_i.mean(0), np.linalg.norm((xyz_i+flow_pd_tmp).mean(0)-xyz_i.mean(0)))
            print('rotation: ',     Rotation.from_matrix(transform[0:3, 0:3]).as_euler("zyx",degrees=True))
            visualize_pcd_multiple(np.concatenate([xyz_i[:,0:3], xyz_j[:,0:3]], axis=0), 
                                   np.concatenate([xyz_i[:,0:3]+flow_pd_tmp, xyz_j[:,0:3]], axis=0), 
                                   np.concatenate([np.zeros((len(xyz_i)))+1, np.zeros((len(xyz_j)))+2], axis=0), 
                                   np.concatenate([np.zeros((len(xyz_i)))+1, np.zeros((len(xyz_j)))+2], axis=0),
                                   num_colors=3,
                                   title = f'eval segment: label {unq}, pair: {pairs[idx]}, flow error {epe3d}, i/j vs i+flow_pd'
                                   )
            # fig = plt.figure()
            # ax1 = plt.subplot(111, projection='3d')
            # ax1.scatter(xyz_i[:, 0], xyz_i[:, 1], xyz_i[:, 2], c='g')
            # ax1.scatter(xyz_j[:, 0], xyz_j[:, 1], xyz_j[:, 2], c='b')
            # ax1.scatter(xyz_i[:, 0]+flow_i[:,0], xyz_i[:, 1]+flow_i[:,1], xyz_i[:, 2]+flow_i[:,2], c='r')
            # # ax2 = plt.subplot(122, projection='3d')
            # # # ax2.sharex(ax1)
            # # # ax2.sharey(ax1)
            # # # ax2.sharez(ax1)
            # # ax1.get_shared_x_axes().join(ax1, ax2)
            # # ax1.get_shared_y_axes().join(ax1, ax2)
            # # ax2.scatter(xyz_i[:, 0], xyz_i[:, 1], xyz_i[:, 2], c='g')
            # # ax2.scatter(xyz_j[:, 0], xyz_j[:, 1], xyz_j[:, 2], c='b')
            # # ax2.scatter(xyz_i[:, 0]+flow_i[:,0], xyz_i[:, 1]+flow_i[:,1], xyz_i[:, 2]+flow_i[:,2], c='r')
            # plt.suptitle(f'result: {pairs[idx]}, {len(xyz_i)}, {len(xyz_j)}, {epe3d}')
            # plt.show()
            # plt.close() 

    return flow
