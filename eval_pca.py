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
import datetime, time
import open3d as o3d
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
import plotly
import torch
import plotly.graph_objs as go
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_loading import collate, ego_motion_compensation, reconstruct_sequence
from utils_eval import calculate_metrics, AverageMeter
from timeit import default_timer as timer
warnings.filterwarnings('ignore')

class Dataset_pca_eval():
    def __init__(self, args):
        self.args = args
        seq_paths = self.meta_data_pca()
        self.seq_paths = seq_paths

        # # randomly sample a subset
        # self.seq_paths = [seq_paths[i] for i in np.random.randint(0, len(seq_paths), (1000))]
        # idxs = [560, 589, 648, 673, 695, 766, 869]
        # self.seq_paths = [self.seq_paths[i] for i in idxs]
        print(f'number of test sequences: {len(self.seq_paths)}')

    def meta_data_pca(self):
        infos = np.loadtxt('assets/configs/datasets/'+self.args.dataset+'/test_info.txt', dtype= str)
        # random.shuffle(infos)
        infos = infos.tolist()
        all_seq_paths = [self.args.root + info for info in infos]
        print(f'infos, total number of test sequences: {len(all_seq_paths)}')

        return all_seq_paths

    def load_data_pca(self, data_path):
        """
        Inut:
            raw_points:     [m, 3]              points before ego-motion compensation
            sd_labels:      [m], dynamic       
            fb_labels:      [m], things, foreground
            inst_labels:    [m]
            time_indice:    [m]
            ego_motion_gt:  [n_frames, 4, 4]
            inst_motion_gt: [k, n_frames, 4, 4]
        """ 
        data = np.load(data_path.replace('test', 'pca'), allow_pickle=True)
        raw_points, time_indice = data['raw_points'], data['time_indice']
        sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
        ego_motion_gt = data['ego_motion_gt']
        scene_flow = data['scene_flow']
        scene_flow_pred = data['scene_flow_pred']
        print(data_path,
            len(np.unique(time_indice)),
            raw_points.shape, time_indice.shape,
            sd_labels.shape, fb_labels.shape,
            ego_motion_gt.shape, 
            scene_flow.shape,
            scene_flow_pred.shape
        )

        assert raw_points.shape[0] == sd_labels.shape[0] == fb_labels.shape[0]==time_indice.shape[0]==scene_flow.shape[0]==scene_flow_pred.shape[0]
        assert len(np.unique(time_indice)) == max(time_indice)+1
        assert max(time_indice)+1==self.args.num_frames

        data = {
            'raw_points': raw_points,
            'time_indice': time_indice,
            'sd_labels': sd_labels,
            'fb_labels': fb_labels,
            'ego_motion_gt': ego_motion_gt,
            'scene_flow': scene_flow,
            'scene_flow_pred': scene_flow_pred,
            'data_path': data_path
        }
        return data

    def __len__(self):
        # print(f'__length__: {len(self.indices_inst_label) * len(self.indices_inst_unlabel)}')
        return len(self.seq_paths)

    def __getitem__(self, idx):
        # wierd bug in PCA paper:
        if '3dd2be428534403ba150a0b60abc6a0a/5dda1ef89d98493aa39a470b21fa4ebe' in self.seq_paths[idx]:
            idx = idx +1
        data = self.load_data_pca(self.seq_paths[idx])
        # print('data path', idx,  data['data_path'])

        # for j in range(1, self.args.num_frames):
        #     point_dst = data['raw_points'][data['time_indice']==0, 0:3]
        #     point_src = data['raw_points'][data['time_indice']==j, 0:3]
        #     flow_gt = data['scene_flow'][data['time_indice']==j]
        #     flow = data['scene_flow_pred'][data['time_indice']==j]
        #     assert len(flow_gt)==len(flow)
        #     # visualize_pcd(np.concatenate([point_src, point_dst, point_src+flow], axis=0), 
        #     #               np.concatenate([np.zeros((len(point_src)))+1, np.zeros((len(point_dst)))+2, np.zeros((len(point_src)))+0], axis=0),
        #     #               num_colors=3, title=f'temporal: src-g, dst-b, src+flow-r: {j} vs {0}')
        #     visualize_pcd(np.concatenate([point_dst, point_src+flow], axis=0), 
        #                   np.concatenate([np.zeros((len(point_dst)))+2, np.zeros((len(point_src)))+1], axis=0),
        #                   num_colors=3, title=f'temporal: src+flow-g, dst-b: {j} vs {0}')

        return data



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
    parser.add_argument('--device_name', type=str, default='cpu',
                        help='Use cuda or not')

    # dataset: 
    parser.add_argument('--dataset', type=str, default='waymo',
                        help='which dataset')
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

    parser.add_argument('--if_ego_pose', default=False, action='store_true',  
                        help='whether to use ego pose or not')

    # ground removal:
    parser.add_argument('--ground_slack', type=float, default=0.3,
                        help='same as the SLIM ICCV21 and PCA ECCV22')
    parser.add_argument('--eval_ground', default=False, action='store_true', 
                        help='remove ground')

    # save / show / debug
    parser.add_argument('--if_save', default=False, action='store_true', 
                        help='save processed data or not')
    parser.add_argument('--savedir', type=str, default='tmp',
                        help='Path to save data (default: ./tmp')
    parser.add_argument('--if_show', default=False, action='store_true', 
                        help='visualize and debug')
    parser.add_argument('--if_verbose', default=False, action='store_true', 
                        help='visualize per segment')
                        
    # parallelization
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers (default: 0)')
    args = parser.parse_args()
    print(f'args: {args}')

    metrics_per_frame = {}
    for metric in ['overall', 'static', 'static_bg', 'static_fg', 'dynamic', 'dynamic_fg']:
        # 0: average over all points over all time steps; i.e., error per point;
        # [1,2, ... , num_frames-1]: average over all points at each time_t, i.e., error per point per time_t;
        # num_frames: average over all points over each scene, i.e., error per point per scene;
        for k in range(0, args.num_frames+1):
            metric_name = metric + f'_{k:d}'
            metrics_per_frame[metric_name] = AverageMeter()
    print('all metrics: ', metrics_per_frame.keys())

    sf_dataset = Dataset_pca_eval(args)
    kwargs = {
        "num_workers": 0,
        "pin_memory": True,
        "collate_fn": collate,
    }
    # https://discuss.pytorch.org/t/is-a-dataset-copied-as-part-of-dataloader-with-multiple-workers/112924
    data_loader = torch.utils.data.DataLoader(
        sf_dataset,
        shuffle=False,
        batch_size=1,
        **kwargs,
    )

    print('start processing at: ', str(datetime.datetime.now()))
    start_time = time.time()
    with torch.no_grad():
        # for k, batch in tqdm(enumerate(data_loader)):
        for k, batch in enumerate(data_loader):
            for kk, data in enumerate(batch): 
                raw_points, time_indice = data['raw_points'], data['time_indice']
                sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
                scene_flow = data['scene_flow']
                flow_seq = data['scene_flow_pred']
                data_path = data['data_path']
                print('data_path: ', data_path)
                # replace the data_path with the baseline result path

                metrics_per_frame = calculate_metrics(args, data, flow_seq, metrics_per_frame)
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
                
        metrics_file = 'metrics_' + args.dataset
        metrics_file += '_' + args.identifier + '.npz'
        np.savez(metrics_file, **metrics_all)
        print(f'save metric file zip: {metrics_file}')
    print('end processing at: ', str(datetime.datetime.now()))
    print('total time (hours): ', (time.time()-start_time)/3600.0)    

