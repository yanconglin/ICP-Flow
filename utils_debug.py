import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as p3d
import seaborn as sns
import scipy
import warnings
import parmap
import time
from operator import mul
from functools import reduce
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, RANSACRegressor
from utils_helper import transform_points
from utils_eval import compute_epe_test
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
warnings.filterwarnings('ignore')

def debug_frame(args, result):
    j = result['j']
    dst = result['dst']
    src = result['src']
    src_ego = transform_points(src,  result['pose'])
    visualize_pcd(np.concatenate([src_ego, dst], axis=0), 
                  np.concatenate([np.zeros((len(src)))+1, np.zeros((len(dst)))+2], axis=0),
                  num_colors=3, title=f'debug input, after ego motion compensation: frame {j}')
    ########################################################################################################
    sd_label = result['sd_label']
    fb_label = result['fb_label']
    flow_gt = result['scene_flow']
    src_label = result['src_label']
    dst_label = result['dst_label']
    flow_direct = result['flow']
    if not args.eval_ground:
        idxs_z = src[:, 2] > args.range_z + args.ground_slack
        src_ego = src_ego[idxs_z]
        src = src[idxs_z]
        src_label = src_label[idxs_z]
        flow_direct = flow_direct[idxs_z]

        sd_label = sd_label[idxs_z]
        fb_label = fb_label[idxs_z]
        flow_gt = flow_gt[idxs_z]

    # overall
    epe3d, accs, accr, outlier, Routlier = compute_epe_test(flow_direct, flow_gt, mask=None)
    print(f"debug frame: {j}/{args.num_frames},  overall, EPE: {epe3d:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")

    # evaluate static
    mask = sd_label==0
    epe3d, accs, accr, outlier, Routlier = compute_epe_test(flow_direct, flow_gt, mask)
    print(f"debug frame: {j}/{args.num_frames},  static, EPE: {epe3d:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")

    # evaluate dynamic
    mask = sd_label==1
    if sum(mask):
        epe3d, accs, accr, outlier, Routlier = compute_epe_test(flow_direct, flow_gt, mask)
        print(f"debug frame: {j}/{args.num_frames}, dynamic, EPE: {epe3d:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")

    # visualize_pcd(np.concatenate([src+flow_direct, dst], axis=0), 
    #                        np.concatenate([np.zeros((len(src)))+1, np.zeros((len(dst)))+2], axis=0), 
    #                        num_colors=3,
    #                        title = f'debug frame {j}: src+flow_pd vs dst'
    #                        )

    label_pd = np.zeros(len(src))-1
    label_pd[sd_label==0] = 1
    label_gt = np.zeros(len(src))-1
    label_gt[sd_label==0] = 2
    visualize_pcd(np.concatenate([src+flow_direct, src+flow_gt], axis=0), 
                           np.concatenate([label_pd, label_gt], axis=0), 
                           num_colors=3,
                           title = f'debug frame {j}: src+flow_pd vs src+flow_gt, static only'
                           )

    label_pd = np.zeros(len(src))-1
    label_pd[sd_label==1] = 1
    label_gt = np.zeros(len(src))-1
    label_gt[sd_label==1] = 2
    visualize_pcd(np.concatenate([src+flow_direct, src+flow_gt], axis=0), 
                           np.concatenate([label_pd, label_gt], axis=0), 
                           num_colors=3,
                           title = f'debug frame {j}: src+flow_pd vs src+flow_gt, dynamic only'
                           )
    # if args.if_verbose:
    #     flow_evaluation(src, dst, src_label, dst_label, 
    #                     flow_direct, flow_gt, 
    #                     result['pose'], result['transformations'],
    #                     pairs=result['pairs']
    #                     )