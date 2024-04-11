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
from utils_flow import flow_estimation_torch
from utils_eval import AverageMeter, calculate_metrics
from utils_loading import collate
from utils_helper import trackers_recursive
from utils_ground import segment_ground
from utils_cluster import cluster_pcd
from utils_debug import debug_frame
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from timeit import default_timer as timer
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

# this loads a pair of frames and gt flow. Note 1). the source has been compensated by ego motion; and 2) ground has been removed.
def dataloader_minimal(data_path):
    data = np.load(data_path)
    pcl_0 = data['pc1']
    pcl_1 = data['pc2']
    valid_0 = data['pc1_flows_valid_idx']
    valid_1 = data['pc2_flows_valid_idx']
    flow_0_1 = data['gt_flow_0_1']
    class_0 = data['pc1_classes']
    class_1 = data['pc2_classes']

    pcl_0 = pcl_0[valid_0]
    pcl_1 = pcl_1[valid_1]
    flow_0_1 = flow_0_1[valid_0]
    class_0 = class_0[valid_0]
    class_1 = class_1[valid_1]
    # print('pc range: ',
    #     pcl_0[:, 0].min(), pcl_0[:, 0].max(), 
    #     pcl_0[:, 1].min(), pcl_0[:, 1].max(), 
    #     pcl_1[:, 0].min(), pcl_1[:, 0].max(), 
    #     pcl_1[:, 1].min(), pcl_1[:, 1].max(), 
    # )
    # print('class 0: ', np.unique(class_0))
    # visualize_pcd(
    #     np.concatenate([pcl_0, pcl_1, pcl_0+flow_0_1], axis=0),
    #     np.concatenate([np.zeros(len(pcl_0))+1, np.zeros(len(pcl_1))+2, np.zeros(len(pcl_0))+0], axis=0),
    #     num_colors=3, 
    #     title=f'zero flow: src-g, dst-b, src+flow-r: {data_path}'
    #     )
    data_dict = {
        'point_src': pcl_0, 
        'point_dst': pcl_1, 
        'scene_flow': flow_0_1,
        'data_path': data_path
    }
    return data_dict


if __name__ == "__main__":

    # Initialization
    random.seed(0)
    np.random.seed(0)
    # fix open3d seed. question mark there: sometimes o3d outputs different results.
    o3d.utility.random.seed(0) # only in open3d>=0.16
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
                        help='points lower than min_height are considered as ground')

    # cluster parameters
    parser.add_argument('--num_clusters', type=int, default=100,
                        help='Number of clusters to keep (default: 100)')
    parser.add_argument('--min_cluster_size', type=int, default=30,
                        help='min_cluster_size (default: 30)')
    parser.add_argument('--epsilon', type=float, default=0.25,
                        help='cluster_selection_epsilon (default: 0.25)')
    parser.add_argument('--if_hdbscan', default=False, action='store_true',  
                        help='use hdbscan')
    parser.add_argument('--if_temporal', default=False, action='store_true',  
                        help='scene flow between any two frames, with temporal alignment')
    parser.add_argument('--if_adjacent', default=False, action='store_true',  
                        help='scene flow between adjacent frames, with temporal alignment')

    # hist parameters
    parser.add_argument('--speed', type=float, default=3.333,
                        help='(default: 120/km/h * 10 Hz)')
    parser.add_argument('--translation_frame', type=float, default=3.333,
                        help='maximal transaltion-xy between two frames')

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

    # ground removal:
    parser.add_argument('--ground_slack', type=float, default=0.3,
                        help='same as the SLIM ICCV21 and PCA ECCV22')
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
    assert args.num_frames==2
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

    files = glob.glob(os.path.join(args.root, args.split+'_fast_flow', '*', '*.npz'))
    print('total files: ', len(files))
    # files.sort()
    random.shuffle(files)
    time_total = 0
    time_cluster = 0
    time_match = 0
    for file in files:
        data = dataloader_minimal(file)
        point_src = data['point_src']
        point_dst = data['point_dst']

        args.translation_frame = args.speed * 2
        # Note: 
        # 0). two successive frames;
        # 1). points have been compendated by ego-motion;
        # 2). the ground has been removed from both;
        time_per_sample = time.time()
        labels = cluster_pcd(args, np.concatenate([point_dst, point_src], axis=0), np.ones(len(point_src)+len(point_dst)).astype(bool))
        label_src = labels[len(point_dst):]
        label_dst = labels[0:len(point_dst)]
        print('cluster time (seconds): ', (time.time()-time_per_sample))
        time_cluster += (time.time()-time_per_sample)
        
        with torch.no_grad():
            time_per_sample = time.time()
            torch.cuda.empty_cache()
            point_src = torch.from_numpy(point_src).float().to(device)
            point_dst = torch.from_numpy(point_dst).float().to(device)
            label_src = torch.from_numpy(label_src).float().to(device)
            label_dst = torch.from_numpy(label_dst).float().to(device)
            pairs, transformations = track(args, point_src, point_dst, label_src, label_dst)
            # print('pairs, transformation: ', pairs.shape, transformations.shape, pairs, transformations)
            flow = flow_estimation_torch(args,
                                src_points=point_src, dst_points=point_dst, 
                                src_labels=label_src, dst_labels=label_dst, 
                                pairs=pairs, transformations=transformations, pose=torch.eye(4)
                                )
            print('match  time (seconds): ', (time.time()-time_per_sample))
            time_match += (time.time()-time_per_sample)

            point_src = point_src.cpu().numpy()
            point_dst = point_dst.cpu().numpy()
            label_src = label_src.cpu().numpy()
            label_dst = label_dst.cpu().numpy()
            flow = flow.cpu().numpy()
                
        if args.if_show:
            # visualize_pcd(np.concatenate([point_src, point_dst], axis=0), 
            #               np.concatenate([np.zeros((len(point_src)))+1, np.zeros((len(point_dst)))+2], axis=0),
            #               num_colors=3, title=f"src vs dst: {data['data_path']}")
            # visualize_pcd(np.concatenate([point_src, point_src+flow], axis=0), 
            #               np.concatenate([np.zeros((len(point_src)))+1, np.zeros((len(point_src)))+2], axis=0),
            #               num_colors=3, title=f"src vs src+flow: {data['data_path']}")
            visualize_pcd(np.concatenate([point_src, point_dst, point_src+flow], axis=0), 
                          np.concatenate([np.zeros(len(point_src))+1, np.zeros(len(point_dst))+2, np.zeros(len(point_src))], axis=0),
                          num_colors=3, title=f"src+flow vs dst: {data['data_path']}")
        if args.if_verbose:
            # if j<3: continue
            result = {
                    'src': point_src[:, 0:3], 
                    'dst': point_dst[:, 0:3], 
                    'src_label': label_src, 
                    'dst_label': label_dst,
                    'pairs': pairs,
                    'transformations': transformations, 
                    'flow': flow,
                    'pose': np.eye(4),
                    'scene_flow': data['scene_flow']
                    } 
            debug_frame(args, result)

        print(f'Processed sample: {data["data_path"]}.')
    print('end processing at: ', str(datetime.datetime.now()))
    


