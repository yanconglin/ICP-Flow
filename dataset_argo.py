import os
import glob
import argparse
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from utils_visualization import visualize_pcd, visualize_pcd_plotly, visualize_pcd_multiple
from utils_helper import transform_points, trackers2labels, trackers_recursive, cum_mm
from utils_cluster import cluster_pcd
import torch

def collate(batch):
    # print('collate batch: ', len(batch), len(batch[0]))
    return batch

class Dataset_argo():
    def __init__(self, args):
        self.args = args
        seq_paths = self.meta_data_pca()
        self.seq_paths = seq_paths
        # self.seq_paths = [seq_paths[i] for i in np.random.randint(0, len(seq_paths), (1000))]
        print(f'number of test sequences: {len(self.seq_paths)}')

        self.background_idxes = [
            CATEGORY_NAME_TO_IDX[cat] for cat in METACATAGORIES["BACKGROUND"]
        ]
        print(f'background idx: {self.background_idxes}')

    def meta_data_pca(self):
        infos = glob.glob(os.path.join(self.args.root, self.args.split+'_zero_flow', '*', '*.npz'))
        infos.sort()
        print(f'infos, total number of test sequences: {len(infos)}')
        return infos

    def load_data_pca(self, data_path):
        # data_info = dict(np.load(data_path))
        # pcl_0 = data_info['pcl_0']
        # pcl_1 = data_info['pcl_1']
        # valid_0 = data_info['valid_0']
        # valid_1 = data_info['valid_1']

        # flow_0_1 = data_info['flow_0_1']
        # flow_1_0 = data_info['flow_1_0']
        # classes_0 = data_info['classes_0']
        # classes_1 = data_info['classes_1']
        # is_ground0 = data_info['is_ground_0']
        # is_ground1 = data_info['is_ground_1']
        # ego_motion = data_info['ego_motion']
        # print('flow info', data_path,
        #     pcl_0.shape, pcl_1.shape, 
        #     valid_0.shape, valid_1.shape,
        #     flow_0_1.shape, flow_1_0.shape,
        #     classes_0.shape, classes_1.shape,
        #     is_ground0.shape, is_ground1.shape, 
        #     ego_motion.shape
        # )

        data2_info = dict(np.load(data_path))
        pcl_0 = data2_info['pc1']
        pcl_1 = data2_info['pc2']
        valid_0 = data2_info['pc1_flows_valid_idx']
        valid_1 = data2_info['pc2_flows_valid_idx']
        flow_0_1 = data2_info['gt_flow_0_1']
        class_0 = data2_info['pc1_classes']
        class_1 = data2_info['pc2_classes']
        ground_0 = data2_info['ground1']
        ground_1 = data2_info['ground2']

        # print('flow info2', data_path,
        #     pcl_0.shape, pcl_1.shape, 
        #     valid_0.shape, valid_1.shape,
        #     flow_0_1.shape,
        #     class_0.shape, class_1.shape,
        #     ground_0.shape, ground_1.shape,
        # )

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

        # argo is 10 HZ, >0,5m/s considered as dynmaic 
        sd_label = np.linalg.norm(flow_0_1, axis=-1)> (0.5 * 0.1)
        fb_label = np.ones((len(pcl_0))).astype(bool)
        for idx in self.background_idxes:
            fb_label[class_0==idx]=False
        fb_label[class_0==-1]=False
        # visualize_pcd(
        #     pcl_0, sd_label+1,
        #     num_colors=3, 
        #     title=f'dynamic, g-static, b-dynamic: {data_path}'
        #     )
        # visualize_pcd(
        #     pcl_0, fb_label+1,
        #     num_colors=3, 
        #     title=f'foreground, g-bg, b-fg: {data_path}'
        #     )
        
        raw_points = np.concatenate([pcl_1,pcl_0], axis=0)
        time_indice = np.concatenate([np.zeros((len(pcl_1))),np.ones((len(pcl_0)))], axis=0)
        ego_motion = np.stack([np.eye(4), np.eye(4)], axis=0)
        scene_flow = np.concatenate([np.zeros((len(pcl_1), 3)), flow_0_1], axis=0)
        # sd and fb labels for pcl1 are not saved, because we evalute on pcl_0 only
        sd_labels = np.concatenate([np.zeros((len(pcl_1))),sd_label], axis=0)
        fb_labels = np.concatenate([np.zeros((len(pcl_1))),fb_label], axis=0)

        data = {
            'raw_points': raw_points,
            'time_indice': time_indice,
            'sd_labels': sd_labels,
            'fb_labels': fb_labels,
            'ego_motion_gt': ego_motion,
            'scene_flow': scene_flow,
            'data_path': data_path
        }

        return data

    def cluster_labels_two(self, data, ego_poses, nonground):
        points_src = []
        points_dst = []
        labels_src = []
        labels_dst = []
        for j in range(1, self.args.num_frames):
            # print(f'calculate scence flow between {j} and {0}')
            point_dst = data['raw_points'][data['time_indice']==0, 0:3]
            point_src = data['raw_points'][data['time_indice']==j, 0:3]
            pose = ego_poses[j]
            point_src_ego = transform_points(point_src,  pose)
            points_tmp = np.concatenate([point_dst, point_src_ego], axis=0)

            nonground_dst = nonground[data['time_indice']==0]
            nonground_src = nonground[data['time_indice']==j]
            nonground_tmp = np.concatenate([nonground_dst, nonground_src], axis=0)
            label_tmp = cluster_pcd(self.args, points_tmp, nonground_tmp)
            label_src = label_tmp[len(point_dst):]
            label_dst = label_tmp[0:len(point_dst)]

            # if j==4:
            #     # visualize_pcd_plotly(point_src_ego[:, 0:3], label_src, num_colors=100)
            #     # visualize_pcd_plotly(point_dst[:, 0:3], label_dst, num_colors=100)
            #     label_src_show = np.zeros((len(point_src)))-1e8
            #     label_src_show[label_src>=0]=1
            #     label_dst_show = np.zeros((len(point_dst)))-1e8
            #     label_dst_show[label_dst>=0]=2
            #     visualize_pcd(np.concatenate([point_src_ego, point_dst], axis=0), np.concatenate([label_src_show, label_dst_show], axis=0),
            #               num_colors=3, title=f'input: src_ego, dst: {j}')
          
            labels_src.append(label_src)
            labels_dst.append(label_dst)
            points_src.append(point_src_ego)
            points_dst.append(point_dst)
        assert len(points_src)==len(points_dst)
        assert len(labels_src)==len(labels_dst)
        assert len(points_src)==len(labels_dst)
        return points_src, points_dst, labels_src, labels_dst

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx):
        # print(f'idx: {idx} / {len(self.seq_paths)}, {self.seq_paths[idx]}')
        data = self.load_data_pca(self.seq_paths[idx])
        ego_poses = data['ego_motion_gt']
        data['ego_poses'] = ego_poses
        nonground = np.ones((len(data['raw_points']))).astype(bool)
        points_src, points_dst, labels_src, labels_dst = self.cluster_labels_two(data, ego_poses, nonground) 
        return data, points_src, points_dst, labels_src, labels_dst


CATEGORY_ID_TO_NAME = {
    -1: 'BACKGROUND',
    0: 'ANIMAL',
    1: 'ARTICULATED_BUS',
    2: 'BICYCLE',
    3: 'BICYCLIST',
    4: 'BOLLARD',
    5: 'BOX_TRUCK',
    6: 'BUS',
    7: 'CONSTRUCTION_BARREL',
    8: 'CONSTRUCTION_CONE',
    9: 'DOG',
    10: 'LARGE_VEHICLE',
    11: 'MESSAGE_BOARD_TRAILER',
    12: 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    13: 'MOTORCYCLE',
    14: 'MOTORCYCLIST',
    15: 'OFFICIAL_SIGNALER',
    16: 'PEDESTRIAN',
    17: 'RAILED_VEHICLE',
    18: 'REGULAR_VEHICLE',
    19: 'SCHOOL_BUS',
    20: 'SIGN',
    21: 'STOP_SIGN',
    22: 'STROLLER',
    23: 'TRAFFIC_LIGHT_TRAILER',
    24: 'TRUCK',
    25: 'TRUCK_CAB',
    26: 'VEHICULAR_TRAILER',
    27: 'WHEELCHAIR',
    28: 'WHEELED_DEVICE',
    29: 'WHEELED_RIDER'
}

CATEGORY_NAME_TO_IDX = {
    v: idx
    for idx, (_, v) in enumerate(sorted(CATEGORY_ID_TO_NAME.items()))
}

SPEED_BUCKET_SPLITS_METERS_PER_SECOND = [0, 0.5, 2.0, np.inf]
ENDPOINT_ERROR_SPLITS_METERS = [0, 0.05, 0.1, np.inf]

BACKGROUND_CATEGORIES = [
    'BOLLARD', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE',
    'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'SIGN', 'STOP_SIGN'
]
PEDESTRIAN_CATEGORIES = [
    'PEDESTRIAN', 'STROLLER', 'WHEELCHAIR', 'OFFICIAL_SIGNALER'
]
SMALL_VEHICLE_CATEGORIES = [
    'BICYCLE', 'BICYCLIST', 'MOTORCYCLE', 'MOTORCYCLIST', 'WHEELED_DEVICE',
    'WHEELED_RIDER'
]
VEHICLE_CATEGORIES = [
    'ARTICULATED_BUS', 'BOX_TRUCK', 'BUS', 'LARGE_VEHICLE', 'RAILED_VEHICLE',
    'REGULAR_VEHICLE', 'SCHOOL_BUS', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
    'TRAFFIC_LIGHT_TRAILER', 'MESSAGE_BOARD_TRAILER'
]
ANIMAL_CATEGORIES = ['ANIMAL', 'DOG']

METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "SMALL_MOVERS": SMALL_VEHICLE_CATEGORIES,
    "LARGE_MOVERS": VEHICLE_CATEGORIES
}

METACATAGORY_TO_SHORTNAME = {
    "BACKGROUND": "BG",
    "PEDESTRIAN": "PED",
    "SMALL_MOVERS": "SMALL",
    "LARGE_MOVERS": "LARGE"
}
