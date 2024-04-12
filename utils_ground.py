import random
import numpy as np
import os
import sys
import hdbscan
import open3d as o3d
from utils_visualization import visualize_pcd
try:
    patchwork_module_path = os.path.join("patchwork-plusplus/build/python_wrapper") # local desktop
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

def segment_ground(args, points):
    nonground_patch = segment_ground_pypatchworkpp(points)
    nonground_thres = segment_ground_thres(args, points) # thresholding as in baselines 
    nonground = np.logical_and(nonground_thres, nonground_patch)
    # visualize_pcd(points, nonground_thres, num_colors=3, title='segment ground thres')
    # visualize_pcd(points, nonground_patch, num_colors=3, title='segment ground patch')
    # visualize_pcd(points, nonground, num_colors=3, title='segment ground')
    return nonground

# same as PCA, ECCV2022 & SLIM ICCV2021
def segment_ground_thres(args, points):
    ground_idxs = points[:, 2]<=args.range_z + args.ground_slack
    labels = np.ones((len(points))).astype(bool)
    labels[ground_idxs] = False
    # # # visualize
    # visualize_pcd(points, labels, num_colors=3, title=f'segment ground: {sum(labels<0)} / {len(points)}')
    return labels

def segment_ground_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    _, ground_idxs = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100, probability=1.0)
    # pcd_rm_ground = pcd.select_by_index(ground_idxs, invert=True)
    labels = np.ones((len(points))).astype(bool)
    labels[ground_idxs] = False
    return labels

def segment_ground_pypatchworkpp(pointcloud):
    # append remission + idxs. 
    # intensity info not used there
    pointcloud = np.concatenate([pointcloud, np.zeros((len(pointcloud), 1)), np.arange(0, len(pointcloud))[:, None]], axis=1)
    assert pointcloud.shape[1]==5
    # print('segment_ground_pypatchworkpp: ', type(pointcloud), pointcloud.shape, pointcloud.dtype, pointcloud[:, -1])

    # Estimate Ground
    # Patchwork++ initialization, to produce deterministic result
    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.enable_RNR = False       # lidar intensity, unclear
    params.sensor_height = 1.723    # lidar position 
    params.min_range = 1.0    
    params.max_range = 64
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    PatchworkPLUSPLUS.estimateGround(pointcloud)
    ground_idx      = PatchworkPLUSPLUS.getGroundIndices()
    nonground_idx   = PatchworkPLUSPLUS.getNongroundIndices()
    labels = np.ones((len(pointcloud))).astype(bool)
    labels[ground_idx] = False

    return labels
