import os
import glob
import re
import argparse
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from utils_visualization import visualize_pcd, visualize_pcd_plotly, visualize_pcd_multiple
from utils_helper import transform_points, trackers2labels, trackers_recursive, cum_mm
from utils_cluster import cluster_pcd
from utils_ground import segment_ground, segment_ground_thres, segment_ground_pypatchworkpp
from utils_ego_motion import egomotion
from utils_loading import ego_motion_compensation, reconstruct_sequence

class Dataset_pca():
    def __init__(self, args):
        self.args = args
        seq_paths = self.meta_data_pca()
        self.seq_paths = seq_paths[0:10]
        print(f'number of test sequences: {len(self.seq_paths)}')

    def meta_data_pca(self):
        infos = np.loadtxt('assets/configs/datasets/'+self.args.dataset+'/'+self.args.split+'_info.txt', dtype= str)
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
        data = np.load(data_path, allow_pickle=True)
        raw_points, time_indice = data['raw_points'], data['time_indice']
        sd_labels, fb_labels = data['sd_labels'], data['fb_labels']
        inst_labels, sem_labels = data['inst_labels'], data['sem_labels']
        ego_motion_gt, inst_motion_gt = data['ego_motion_gt'], data['bbox_tsfm']
        # print(data_path,
        #     len(np.unique(time_indice)),
        #     raw_points.shape, time_indice.shape,
        #     sd_labels.shape, fb_labels.shape,
        #     inst_labels.shape, sem_labels.shape, 
        #     ego_motion_gt.shape, inst_motion_gt.shape
        # )

        assert raw_points.shape[0] == sd_labels.shape[0] == fb_labels.shape[0]==inst_labels.shape[0]==time_indice.shape[0]
        assert ego_motion_gt.shape[0] == len(np.unique(time_indice))
        assert inst_motion_gt.shape[1] == ego_motion_gt.shape[0]
        assert len(np.unique(time_indice)) == max(time_indice)+1
        assert max(time_indice)+1==self.args.num_frames

        # # # crop the scene (x-y dimensions only)
        idxs_xy = np.logical_and(np.abs(raw_points[:,0]) < self.args.range_x, np.abs(raw_points[:,1]) < self.args.range_y)
        raw_points, time_indice = raw_points[idxs_xy], time_indice[idxs_xy]
        sd_labels, fb_labels = sd_labels[idxs_xy], fb_labels[idxs_xy]
        inst_labels = inst_labels[idxs_xy]

        # # # resonctruct scene flow
        points_ego = ego_motion_compensation(raw_points, time_indice, ego_motion_gt)
        points_full = reconstruct_sequence(points_ego, time_indice, inst_labels, inst_motion_gt, len(np.unique(time_indice)))
        scene_flow = points_full - raw_points  #  flow vector from frame t to frame 1. 

        # # # # # # # # # visualize:
        # for k in range(max(time_indice)-1, max(time_indice)):
        #     pcd_0 = raw_points[time_indice==0, 0:3]
        #     pcd_t = raw_points[time_indice==k, 0:3]
        #     pcd_ego_t = points_ego[time_indice==k, 0:3]
        #     pcd_full_t = points_full[time_indice==k, 0:3]
        #     flow_t = scene_flow[time_indice==k, 0:3]
        #     sd_labels_t = sd_labels[time_indice==k] 
        #     fb_labels_t = fb_labels[time_indice==k] 

        #     visualize_pcd(pcd_t, sd_labels_t,
        #         num_colors=3,
        #         title=f'visualize {k} th example and sd labels'
        #         )
        #     visualize_pcd(pcd_t, fb_labels_t,
        #         num_colors=3,
        #         title=f'visualize {k} th example and fb labels'
        #         )

        #     visualize_pcd(
        #         np.concatenate([pcd_0, pcd_ego_t], axis=0),
        #         np.concatenate([np.zeros(len(pcd_0)), np.zeros(len(pcd_t))+1], axis=0),
        #         num_colors=3,
        #         title=f'{k}: frame 0 vs frame t ego'
        #         )
        #     visualize_pcd(
        #         np.concatenate([pcd_0, pcd_t + flow_t], axis=0),
        #         np.concatenate([np.zeros(len(pcd_0)), np.zeros(len(pcd_t))+1], axis=0),
        #         num_colors=3,
        #         title=f'{k}: frame 0 vs frame t + flow_t'
        #         )

        data = {
            'raw_points': raw_points,
            'time_indice': time_indice,
            'sd_labels': sd_labels,
            'fb_labels': fb_labels,
            'ego_motion_gt': ego_motion_gt,
            'scene_flow': scene_flow,
            'data_path': data_path
        }

        return data

    def ego_motion_estimation(self, data):
        # egomotion estimation
        for folder in ['train', 'val', 'test']:
            if folder in data['data_path']:
                path_pose = data['data_path'].replace(folder, folder+'_pose')
                break
        if os.path.isfile(path_pose):
            ego_poses = np.load(path_pose, allow_pickle=True)['ego_motion']

        else:
            # unfortunately, poses are still not deterministic
            egomotioner = egomotion(self.args)
            for i in range(0, self.args.num_frames): #[0, 1], [0, 2], [0, 3],...,[0, N-1]
                src = data['raw_points'][data['time_indice']==i, 0:3]
                _ = egomotioner.register_frame(src, i)
                # print(f'pose: {i} th frame, {pose_i}')
            assert len(egomotioner.poses)==self.args.num_frames
            ego_poses = egomotioner.poses
            if not os.path.exists(os.path.dirname(path_pose)):
                os.makedirs(os.path.dirname(path_pose), exist_ok=True)
            np.savez_compressed(path_pose, ego_motion = ego_poses)

        # # # visualize ego-motion estimation
        # for j in range(3, num_frames): #[0, 1], [0, 2], [0, 3],...,[0, N-1]
        #     # print(f'estimate ego motion for {j} th frame')
        #     seq_0_raw = data['raw_points'][time_indice==0, 0:3]
        #     seq_j_raw = data['raw_points'][time_indice==j, 0:3]
        #     pose_j_gt = data['ego_motion_gt'][j]
        #     pose_j = ego_poses[j]
        #     seq_j_ego = transform_points(seq_j_raw, pose_j)
        #     print(f'pose for frame {j} pd: {pose_j}')
        #     print(f'               {j} gt: {pose_j_gt}')
        #     visualize_pcd(np.concatenate([seq_j_ego, seq_0_raw], axis=0), 
        #                   np.concatenate([np.zeros((len(seq_j_ego)))+1, np.zeros((len(seq_0_raw)))+2], axis=0),
        #                   num_colors=3, title=f'ego pose estimation: {j}')
        return ego_poses
    
    def ground_removal(self, data):
        nonground = []
        for j in range(0, self.args.num_frames):
            points_tmp = data['raw_points'][data['time_indice']==j, 0:3]
            nonground_tmp = segment_ground(self.args, points_tmp)
            nonground.append(nonground_tmp)
        nonground = np.hstack(nonground)
        # print(f'ground removel: {len(data["raw_points"])} and {len(nonground)}')
        assert len(data['raw_points'])==len(nonground)
        return nonground


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

    def pad_segments(self, points, labels):
        segs = []
        paddings = []
        unqs = np.unique(labels)
        for unq in unqs:
            seg = points[labels==unq]
            padding = np.zeros((self.args.max_points, 1)) + 1.0
            if len(seg) > self.args.max_points:
                sample_idxs = np.random.choice(len(seg), self.args.max_points)
                seg = seg[sample_idxs, :]
            elif len(seg) < self.args.max_points:
                padding[len(seg):] = 0.0
                seg = np.concatenate([seg, np.zeros((self.args.max_points-len(seg), 3))+1e8], axis=0)
            else: 
                pass
            assert len(seg)==self.args.max_points
            assert len(seg)==len(padding)
            segs.append(seg)
            paddings.append(padding)
        segs = np.stack(segs, axis=0)
        paddings = np.stack(paddings, axis=0)
        # print('segs: ', segs.shape, paddings.shape)
        return np.concatenate([segs, paddings], axis=-1)

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx):
        # print(f'idx: {idx} / {len(self.seq_paths)}, {self.seq_paths[idx]}')
        data = self.load_data_pca(self.seq_paths[idx])
        nonground = self.ground_removal(data)
        if self.args.if_kiss_icp:
            ego_poses = self.ego_motion_estimation(data)
        else:
            ego_poses = data['ego_motion_gt']
        data['ego_poses'] = ego_poses

        points_src, points_dst, labels_src, labels_dst = self.cluster_labels_two(data, ego_poses, nonground) 

        return data, points_src, points_dst, labels_src, labels_dst
