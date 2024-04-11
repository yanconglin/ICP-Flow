import numpy as np
import random
import glob
import warnings
import parmap
import os
import itertools
import shutil
import argparse
import copy
import scipy
import sys
import yaml
import json
import datetime, time
import open3d as o3d
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
from utils_visualization import visualize_pcd, visualize_pcd_plotly, visualize_pcd_multiple
from timeit import default_timer as timer
warnings.filterwarnings('ignore')

def crop_data(args, data, pred):
    # # crop the scene (x-y) and remove the ground (z)
    # MAKE SURE THIS IS THE SAME as Dynamic 3D Scene Analysis by Point Cloud Accumulation, ECCV2022
    raw_points, time_indice = data['raw_points'], data['time_indice']
    sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
    ego_motion_gt = data['ego_motion_gt']
    scene_flow = data['scene_flow']
    data_path = data['data_path']

    idxs_xy = np.logical_and(np.abs(raw_points[:,0]) < args.range_x, np.abs(raw_points[:,1]) < args.range_y)
    if args.eval_ground:
        idxs_xyz = idxs_xy
    else:
        idxs_z = raw_points[:,2] > args.range_z + args.ground_slack
        idxs_xyz = np.logical_and(idxs_xy, idxs_z)
    # print('crop data: ', raw_points.shape, sd_labels.shape, fb_labels.shape, scene_flow.shape, time_indice.shape, pred.shape)
    raw_points_crop =raw_points[idxs_xyz]
    time_indice_crop =time_indice[idxs_xyz]
    sd_labels_crop =sd_labels[idxs_xyz]
    fb_labels_crop =fb_labels[idxs_xyz]
    scene_flow_crop =scene_flow[idxs_xyz]
    pred_crop =pred[idxs_xyz]

    # # # # # # # # # # # visualize:
    # visualize_pcd_multiple(
    #     raw_points, raw_points_crop, sd_labels, sd_labels_crop,
    #     num_colors=3,
    #     title=f'crop, sd labels'
    #     )
    data_crop = {
        'raw_points': raw_points_crop,
        'time_indice': time_indice_crop,
        'sd_labels': sd_labels_crop,
        'fb_labels': fb_labels_crop,
        'ego_motion_gt': ego_motion_gt,
        'scene_flow': scene_flow_crop,
        'data_path': data_path
    }

    return  data_crop, pred_crop

def average_meter(errors, nums):
    """
    Calculate the mean error over all flow vectors averaged over all frames.
    Same as "Dynamic 3D Scene Analysis by Point Cloud Accumulation, ECCV22"

    """
    assert len(errors)==len(nums)
    total_error = 0
    total_num = 0
    for (error, num) in zip(errors, nums):
        total_error += error * num
        total_num += num

    errors_mean = total_error/total_num
    return errors_mean
        

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.num = 0
        self.epe_sum = 0.0
        self.epe_avg = 0.0

        self.accs_sum = 0.0
        self.accs_avg = 0.0

        self.accr_sum = 0.0
        self.accr_avg = 0.0

        self.outlier_sum = 0.0
        self.outlier_avg = 0.0

        self.Routlier_sum = 0.0
        self.Routlier_avg = 0.0

        self.num_data = []
        self.epe_data = []
        self.accs_data = []
        self.accr_data = []
        self.outlier_data = []
        self.Routlier_data = []

    # epe, accs, accr, outlier, Routlier
    def update(self, epe, accs, accr, outlier, Routlier, num):
        self.num += num
        self.num_data.append(num)

        self.epe_sum += epe * num
        self.epe_avg = self.epe_sum / self.num
        self.epe_data.append(epe)

        self.accs_sum += accs * num
        self.accs_avg = self.accs_sum / self.num
        self.accs_data.append(accs)

        self.accr_sum += accr * num
        self.accr_avg = self.accr_sum / self.num
        self.accr_data.append(accr)

        self.outlier_sum += outlier * num
        self.outlier_avg = self.outlier_sum / self.num
        self.outlier_data.append(outlier)

        self.Routlier_sum += Routlier * num
        self.Routlier_avg = self.Routlier_sum / self.num
        self.Routlier_data.append(Routlier)

def compute_epe_test(flow_pred, flow_gt, mask=None):
    """
    Compute EPE, accuracy and number of outliers.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Flow
    assert flow_gt.shape[-1]==3
    assert flow_pred.shape[-1]==3

    if mask is not None:
        flow_gt = flow_gt[mask > 0]
        flow_pred = flow_pred[mask > 0]

    # EPE
    epe3d_per_point = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe3d = epe3d_per_point.mean()
    # print('compute epe3d: ', epe3d)

    sf_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err_per_point = epe3d_per_point / (sf_norm + 1e-20)

    acc3d_strict_per_point = (np.logical_or(epe3d_per_point < 0.05, relative_err_per_point < 0.05)).astype(np.float32)
    acc3d_strict = acc3d_strict_per_point.mean()

    acc3d_relax_per_point = (np.logical_or(epe3d_per_point < 0.1, relative_err_per_point < 0.1)).astype(np.float32)
    acc3d_relax = acc3d_relax_per_point.mean()

    outlier_per_point = (np.logical_or(epe3d_per_point > 0.3, relative_err_per_point > 0.1)).astype(np.float32)
    outlier = outlier_per_point.mean()

    Routlier_per_point = (np.logical_and(epe3d_per_point > 0.3, relative_err_per_point > 0.3)).astype(np.float32)
    Routlier = Routlier_per_point.mean()

    return epe3d, acc3d_strict, acc3d_relax, outlier, Routlier


def calculate_metrics(args, data, flow_seq, metrics_per_frame):
    if args.eval_ground:
        pass
    else:
        data, flow_seq = crop_data(args, data, flow_seq)

    raw_points, time_indice = data['raw_points'], data['time_indice']
    sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
    scene_flow = data['scene_flow']

    ###################### evaluate per frame ############################################
    time_indice = data['time_indice']
    num_frames = len(np.unique(time_indice))
    assert num_frames==args.num_frames
    sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
    for j in range(1, num_frames):
        src_j = raw_points[time_indice==j, 0:3]
        dst_0 = raw_points[time_indice==0, 0:3] # frame 0
        flow_gt_j = scene_flow[time_indice==j, 0:3]
        flow_pd_j = flow_seq[time_indice==j, 0:3]
        sd_labels_j = sd_labels[time_indice==j]
        fb_labels_j = fb_labels[time_indice==j]

        # error per point, at a particular time step t
        # overall
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_pd_j, flow_gt_j, mask=None)
        print(f"frame: {j:02d}, overall, EPE3D: {epe:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")
        metric_name = 'overall' + f'_{j:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, len(flow_pd_j))

        # evaluate static
        mask = sd_labels_j==0
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_pd_j, flow_gt_j, mask)
        print(f"frame: {j:02d},  static, EPE3D: {epe:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")
        metric_name = 'static' + f'_{j:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

        # evaluate static background
        mask = np.logical_and(sd_labels_j==0, fb_labels_j==0)
        if sum(mask):
            epe, accs, accr, outlier, Routlier = compute_epe_test(flow_pd_j, flow_gt_j, mask)
            metric_name = 'static_bg' + f'_{j:d}'
            assert metric_name in metrics_per_frame.keys()
            metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

        # evaluate static foreground
        mask = np.logical_and(sd_labels_j==0, fb_labels_j==1)
        if sum(mask):
            epe, accs, accr, outlier, Routlier = compute_epe_test(flow_pd_j, flow_gt_j, mask)
            metric_name = 'static_fg' + f'_{j:d}'
            assert metric_name in metrics_per_frame.keys()
            metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

        # evaluate dynamic
        mask = sd_labels_j==1
        if sum(mask):
            epe, accs, accr, outlier, Routlier = compute_epe_test(flow_pd_j, flow_gt_j, mask)
            print(f"frame: {j:02d}, dynamic, EPE3D: {epe:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")
            metric_name = 'dynamic' + f'_{j:d}'
            assert metric_name in metrics_per_frame.keys()
            metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))
        else:
            epe, accs, accr, outlier = 0.0, 0.0, 0.0, 0.0  # simply to make the output doc coherent. THESE VALUES ARE NOT USED IN EVALUATION!
            print(f"frame: {j:02d}, dynamic, EPE3D: {epe:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")

        # evaluate dynamic fg
        mask = np.logical_and(sd_labels_j==1, fb_labels_j==1)
        if sum(mask):
            epe, accs, accr, outlier, Routlier = compute_epe_test(flow_pd_j, flow_gt_j, mask)
            # print(f"file: {i}/{len(data_paths)}-{j}, dynamic_fg, EPE3D: {epe:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Rutlier: {Routlier:.4f}")
            metric_name = 'dynamic_fg' + f'_{j:d}'
            assert metric_name in metrics_per_frame.keys()
            metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))
        else:
            epe3d, accs, accr, outlier, Routlier = 0.0, 0.0, 0.0, 0.0, 0.0  # simply to make the output doc coherent. THESE VALUES ARE NOT USED IN EVALUATION!
            # print(f"file: {i}/{len(data_paths)}-{j}, dynamic_fg, EPE3D: {epe:.4f}, ACC3DS: {accs:.4f}, ACC3DR: {accr:.4f}, Outlier: {outlier:.4f}, Routlier: {Routlier:.4f}")

    ##################################################################################################################################################
    # evaluate all: Calculate error per point over all time steps
    valid = time_indice>0 # oops, do not take time=0 into consideration!
    flow_seq_ = flow_seq[valid]
    scene_flow_ = scene_flow[valid]
    sd_labels_ = sd_labels[valid]
    fb_labels_ = fb_labels[valid]

    epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask=None)
    metric_name = 'overall' + f'_{0:d}'
    assert metric_name in metrics_per_frame.keys()
    metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, len(flow_seq))

    # evaluate static
    mask = sd_labels_==0
    epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
    metric_name = 'static' + f'_{0:d}'
    assert metric_name in metrics_per_frame.keys()
    metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

    # evaluate static background
    mask = np.logical_and(sd_labels_==0, fb_labels_==0)
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'static_bg' + f'_{0:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

    # evaluate static foreground
    mask = np.logical_and(sd_labels_==0, fb_labels_==1)
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'static_fg' + f'_{0:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

    # evaluate dynamic
    mask = sd_labels_==1
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'dynamic' + f'_{0:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

    # dynamic foreground 
    mask = np.logical_and(sd_labels_==1, fb_labels_==1)
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'dynamic_fg' + f'_{0:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, sum(mask))

    ##################################################################################################################################################
    # evaluate all: Calculate error per point per scene
    valid = time_indice>0 
    flow_seq_ = flow_seq[valid]
    scene_flow_ = scene_flow[valid]
    sd_labels_ = sd_labels[valid]
    fb_labels_ = fb_labels[valid]

    epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask=None)
    metric_name = 'overall' + f'_{args.num_frames:d}'
    assert metric_name in metrics_per_frame.keys()
    metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, 1)

    # evaluate static
    mask = sd_labels_==0
    epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
    metric_name = 'static' + f'_{args.num_frames:d}'
    assert metric_name in metrics_per_frame.keys()
    metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, 1)

    # evaluate static background
    mask = np.logical_and(sd_labels_==0, fb_labels_==0)
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'static_bg' + f'_{args.num_frames:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, 1)

    # evaluate static foreground
    mask = np.logical_and(sd_labels_==0, fb_labels_==1)
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'static_fg' + f'_{args.num_frames:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, 1)

    # evaluate dynamic
    mask = sd_labels_==1
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'dynamic' + f'_{args.num_frames:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, 1)

    # dynamic foreground 
    mask = np.logical_and(sd_labels_==1, fb_labels_==1)
    if sum(mask):
        epe, accs, accr, outlier, Routlier = compute_epe_test(flow_seq_, scene_flow_, mask)
        metric_name = 'dynamic_fg' + f'_{args.num_frames:d}'
        assert metric_name in metrics_per_frame.keys()
        metrics_per_frame[metric_name].update(epe, accs, accr, outlier, Routlier, 1)

    return metrics_per_frame