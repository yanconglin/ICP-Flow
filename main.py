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
import sys
import yaml
import json
import gc
import datetime, time
import open3d as o3d
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
import plotly
from tqdm import tqdm
import torch
import plotly.graph_objs as go
from dataset_pca import Dataset_pca
from dataset_argo import Dataset_argo
from utils_track import track
from utils_flow import flow_estimation
from utils_eval import AverageMeter, calculate_metrics
from utils_loading import collate
from utils_debug import debug_frame
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly
from timeit import default_timer as timer
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

if __name__ == "__main__":

    # Initialization
    random.seed(0)
    np.random.seed(0)
    o3d.utility.random.seed(0)    # fix open3d seed, only in open3d>=0.16.
    multiprocessing.set_start_method('forkserver') # or 'spawn'?
    
    # Parse hyperparameters
    parser = argparse.ArgumentParser(description='SceneFlow')

    parser.add_argument('--identifier', type=str, default='run',
                        help='identify which run')
    parser.add_argument('--if_gpu', default=False, action='store_true',  
                        help='whether to use gpu or not')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='gpu_idx (default: 0)')

    # ego motion estimation
    parser.add_argument('--config', type=str, default='config_kiss_icp.yaml',
                        help='config file for egomotion estimator (a copy from kiss_icp)')
    parser.add_argument('--if_kiss_icp', default=False, action='store_true',  
                        help='use kiss_icp or gt for ego motion compensation')

    # dataset: 
    parser.add_argument('--dataset', type=str, default='waymo',
                        help='which dataset')
    parser.add_argument('--split', type=str, default='test',
                        help='split: train/val/test')
    parser.add_argument('--root', type=str, default='/mnt/Data/Dataset/eth_scene_flow/compressed/waymo/val',
                        help='Path to dataset')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames per file (default: 5)')
    parser.add_argument('--range_x', type=float, default=32,
                        help='crop the scene, +- range-x')
    parser.add_argument('--range_y', type=float, default=32,
                        help='crop the scene, +- range-y')
    parser.add_argument('--range_z', type=float, default=0.0,
                        help='ground<=range_z, same as PCA ECCV22')

    # cluster parameters
    parser.add_argument('--num_clusters', type=int, default=100,
                        help='Number of clusters to keep (default: 100)')
    parser.add_argument('--min_cluster_size', type=int, default=30,
                        help='min_cluster_size (default: 30)')
    parser.add_argument('--epsilon', type=float, default=0.25,
                        help='cluster_selection_epsilon (default: 0.25)')
    parser.add_argument('--if_hdbscan', default=False, action='store_true',  
                        help='use hdbscan')

    # hist parameters
    parser.add_argument('--speed', type=float, default=3.333,
                        help='(default: 120/km/h * 10 Hz)')
    parser.add_argument('--translation_frame', type=float, default=3.333,
                        help='maximal transaltion-xy between two frames')
    parser.add_argument('--chunk_size', type=int, default=50,
                        help='useful to prevent memeory overflow (default: 50)')

    # icp parameters
    parser.add_argument('--thres_dist', type=float, default=0.1,
                        help='threshold that determines a pair of inlier')
    parser.add_argument('--max_points', type=int, default=10000,
                        help='max number of points')

    # sanity check
    parser.add_argument('--thres_box', type=float, default=0.1,
                        help='bbox threshold to reject pairs')

    # filter unreliable matches
    parser.add_argument('--thres_error', type=float, default=0.1,
                        help='threshold to reject unreliable matches')
    parser.add_argument('--thres_iou', type=float, default=0.1,
                        help='threshold to reject unreliable matches')
    # parser.add_argument('--thres_rot', type=float, default=0.1,
    #                     help='threshold to reject unreliable matches')

    # ground removal:
    parser.add_argument('--ground_slack', type=float, default=0.3,
                        help='same as SLIM ICCV21 and PCA ECCV22')
    parser.add_argument('--eval_ground', default=False, action='store_true', 
                        help='remove ground')

    # save / show / debug
    parser.add_argument('--if_save', default=False, action='store_true', 
                        help='save processed data or not')
    parser.add_argument('--if_show', default=False, action='store_true', 
                        help='visualize and debug')
    parser.add_argument('--if_verbose', default=False, action='store_true', 
                        help='visualize per segment')
                        
    # parallelization
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers (default: 1)')

    args = parser.parse_args()
    assert args.num_workers<=multiprocessing.cpu_count()
    assert args.batch_size==1
    args.identifier = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    print('start processing at: ', str(datetime.datetime.now()))
    print(f'args: {args}')

    torch.manual_seed(0)
    if args.if_gpu:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count()==1
        device_name = f'cuda:{torch.cuda.current_device()}'
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        torch.cuda.empty_cache()
        gc.collect()
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print(f"Let's use {torch.cuda.device_count()}, {torch.cuda.get_device_name()} GPU(s)!")
    else:
        print("CUDA is not available")
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f'device: {device}')

    if args.dataset in ['waymo', 'nuscene']:
        sf_dataset = Dataset_pca(args)
    elif args.dataset in ['argo']:
        sf_dataset = Dataset_argo(args)
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate,
    }
    # https://discuss.pytorch.org/t/is-a-dataset-copied-as-part-of-dataloader-with-multiple-workers/112924
    data_loader = torch.utils.data.DataLoader(
        sf_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        **kwargs,
    )

    metrics_per_frame = {}
    for metric in ['overall', 'static', 'static_bg', 'static_fg', 'dynamic', 'dynamic_fg']:
        # 0: average over all points over all frames (timesteps); i.e., error per point;
        # [1,2, ... , num_frames-1]: average over all points at each frame t (time_t), i.e., error per point per time_t;
        # num_frames: average over all points over each scene, i.e., error per point per scene;
        for k in range(0, args.num_frames+1):
            metric_name = metric + f'_{k:d}'
            metrics_per_frame[metric_name] = AverageMeter()
    print('all metrics: ', metrics_per_frame.keys())

    start_time = time.time()
    for k, batch in enumerate(data_loader):
        data, points_src, points_dst, labels_src, labels_dst = batch[0]
        ego_poses = data['ego_poses']
        pairs_t = []
        transformations_t = []
        for i, (point_src, point_dst, label_src, label_dst) in enumerate(zip(points_src, points_dst, labels_src, labels_dst)):
            # print('i: ', i)
            # visualize_pcd_plotly(point_src[:, 0:3], label_src, num_colors=100)
            # visualize_pcd_plotly(point_dst[:, 0:3], label_dst, num_colors=100)
            # visualize_pcd(np.concatenate([point_src, point_dst], axis=0), 
            #               np.concatenate([np.zeros((len(point_src)))+1, np.zeros((len(point_dst)))+2], axis=0),
            #               num_colors=3, title=f'input: src vs dst: {i+1}-th pair')
            # visualize_pcd(np.concatenate([point_src, point_dst], axis=0), 
            #               np.concatenate([label_src, label_dst], axis=0),
            #               num_colors=3, title=f'input: src vs dst: {i+1}-th pair')
            # update translation_frame
            args.translation_frame = max(args.speed * 1, np.linalg.norm(ego_poses[i+1][0:3, -1])) * 2

            # this demo uses GPUs for ICP calculating. 
            with torch.no_grad():
                torch.cuda.empty_cache()
                point_src = torch.from_numpy(point_src).float().to(device)
                point_dst = torch.from_numpy(point_dst).float().to(device)
                label_src = torch.from_numpy(label_src).float().to(device)
                label_dst = torch.from_numpy(label_dst).float().to(device)
                pairs, transformations = track(args, point_src, point_dst, label_src, label_dst)
            # back to CPU
            pairs = pairs.cpu().numpy()
            transformations = transformations.cpu().numpy()

            pairs_t.append(pairs)
            transformations_t.append(transformations)

        flows = []
        flows.append(np.zeros((len(points_dst[0]), 3))) # append zero flows for frame 0.

        for j in range(1, args.num_frames):
            pairs, transformations = pairs_t[j-1], transformations_t[j-1]
            # assigning the same labels to corresponding instances; useful for visualization 
            # tracker_src, tracker_dst = trackers2labels(label_src, labels_dst[0], pairs)
            point_dst = data['raw_points'][data['time_indice']==0, 0:3]
            point_src = data['raw_points'][data['time_indice']==j, 0:3]
            # if not adjacent
            label_dst = labels_dst[j-1]
            label_src = labels_src[j-1]

            flow = flow_estimation(args,
                                src_points=point_src, dst_points=point_dst, 
                                src_labels=label_src, dst_labels=label_dst, 
                                pairs=pairs, transformations=transformations, pose=ego_poses[j]
                                )

            if args.if_show:
                visualize_pcd(np.concatenate([point_src+flow, point_dst], axis=0), 
                              np.concatenate([np.zeros((len(point_src)))+1, np.zeros((len(point_dst)))+2], axis=0),
                              num_colors=3, title=f'temporal: src+flow vs dst: {j} vs {0}')

            if args.if_verbose:
                result = {
                        'j': j, 
                        'src': point_src[:, 0:3], 
                        'dst': point_dst[:, 0:3], 
                        'src_label': label_src, 
                        'dst_label': label_dst,
                        'pairs': pairs,
                        'transformations': transformations, 
                        'flow': flow,
                        'pose': ego_poses[j],
                        'sd_label': data['sd_labels'][data['time_indice']==j],
                        'fb_label': data['fb_labels'][data['time_indice']==j],
                        'scene_flow': data['scene_flow'][data['time_indice']==j, 0:3],
                        } 
                debug_frame(args, result)

            flows.append(flow)

        flows = np.vstack(flows)
        print(f'Processed sample {k}/{len(sf_dataset.seq_paths)}, {data["data_path"]}.')
        if args.if_save:
            for folder in ['train', 'val', 'test']:
                if folder in data['data_path']:
                    path_flow = data['data_path']
                    if args.if_kiss_icp:
                        path_flow = path_flow.replace(folder, os.path.join(folder+'_icp_flow'))
                    else:
                        path_flow = path_flow.replace(folder, os.path.join(folder+'_icp_flow_ego'))

                    if args.if_adjacent:
                        path_flow = path_flow.replace(folder, os.path.join(folder+'_adjacent'))
                    elif args.if_temporal:
                        path_flow = path_flow.replace(folder, os.path.join(folder+'_temporal'))
                    else: 
                        pass
                    break
            assert path_flow!=data['data_path']
            if not os.path.exists(os.path.dirname(path_flow)):
                os.makedirs(os.path.dirname(path_flow), exist_ok=True)
            np.savez_compressed(path_flow, 
                                scene_flow = flows, 
                                ego_motion = ego_poses,
                                )
        metrics_per_frame = calculate_metrics(args, data, flows, metrics_per_frame)

    ##################################################################################################################################################
    print(f'################# Results over the entire dataset #####################################')
    for k in range(0, args.num_frames+1):
        for metric in ['overall', 'static', 'static_bg', 'static_fg', 'dynamic', 'dynamic_fg']:
            metric_name = metric + f'_{k:d}'
            print(f'{metric_name:12}, EPE3D: {metrics_per_frame[metric_name].epe_avg:.6f}, \
                  ACC3DS: {metrics_per_frame[metric_name].accs_avg:.6f}, \
                  ACC3DR: {metrics_per_frame[metric_name].accr_avg:.6f}, \
                  Outlier: {metrics_per_frame[metric_name].outlier_avg:.6f}, \
                  Routlier: {metrics_per_frame[metric_name].Routlier_avg:.6f}.')

    if args.if_save:
        metrics_all = {}
        for k in range(0, args.num_frames+1):
            for metric in ['overall', 'static', 'static_bg', 'static_fg', 'dynamic', 'dynamic_fg']:
                metrics_all['EPE3D' + metric + f'_{k:d}'] = metrics_per_frame[metric + f'_{k:d}'].epe_data,
                metrics_all['ACC3DS_' + metric + f'_{k:d}'] = metrics_per_frame[metric + f'_{k:d}'].accs_data,
                metrics_all['ACC3DR_' + metric + f'_{k:d}'] = metrics_per_frame[metric + f'_{k:d}'].accr_data,
                metrics_all['OUTLIER_' + metric + f'_{k:d}'] = metrics_per_frame[metric + f'_{k:d}'].outlier_data,
                metrics_all['ROUTLIER_' + metric + f'_{k:d}'] = metrics_per_frame[metric + f'_{k:d}'].Routlier_data,
                
        metrics_file = 'metrics_' + args.dataset + '_' + args.split
        if args.if_kiss_icp: metrics_file += f'_icp'
        metrics_file += '_' + args.identifier + '.npz'
        np.savez(metrics_file, **metrics_all)
        print(f'save metric file zip: {metrics_file}')
    print('end processing at: ', str(datetime.datetime.now()))
    print('total time (hours): ', (time.time()-start_time)/3600.0)
    


