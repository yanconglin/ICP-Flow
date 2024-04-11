import random
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_helper import transform_points
from utils_ground import segment_ground_thres, segment_ground_pypatchworkpp

from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import KISSConfig
from kiss_icp.deskew import get_motion_compensator
from kiss_icp.mapping import get_voxel_hash_map
from kiss_icp.preprocess import get_preprocessor
from kiss_icp.registration import register_frame
from kiss_icp.threshold import get_threshold_estimator
from kiss_icp.voxelization import voxel_down_sample

class egomotion:
    def __init__(self, args):
        random.seed(0)
        np.random.seed(0)
        config = self.load_config(args.config)
        self.args = args
        self.poses = []
        self.config = config
        self.compensator = get_motion_compensator(config)
        self.adaptive_threshold = get_threshold_estimator(self.config)
        self.local_map = get_voxel_hash_map(self.config)
        self.preprocess = get_preprocessor(self.config)

    def load_config(self, config_file) -> KISSConfig:
        """Load configuration from an Optional yaml file. Additionally, deskew and max_range can be
        also specified from the CLI interface"""

        config = KISSConfig(config_file=config_file)

        # Check if there is a possible mistake
        if config.data.max_range < config.data.min_range:
            print("[WARNING] max_range is smaller than min_range, settng min_range to 0.0")
            config.data.min_range = 0.0

        # Use specified voxel size or compute one using the max range
        if config.mapping.voxel_size is None:
            config.mapping.voxel_size = float(config.data.max_range / 100.0)

        return config

    def register_frame(self, frame, timestamps):
        # print('timestamps: ', np.unique(timestamps))
        # Apply motion compensation
        frame = self.compensator.deskew_scan(frame, self.poses, np.zeros((len(frame)))+timestamps)
        # print('deskew frame: ', frame.shape)

        # Preprocess the input cloud
        frame = self.preprocess(frame)
        # print('preprocess frame: ', frame.shape)

        # Voxelize
        source, frame_downsample = self.voxelize(frame)
        # print('preprocess frame: ', source.shape, frame_downsample.shape)

        # Get motion prediction and adaptive_threshold
        sigma = self.get_adaptive_threshold()
        # print('sigma: ', sigma)

        # Compute initial_guess for ICP
        prediction = self.get_prediction_model()
        last_pose = self.poses[-1] if self.poses else np.eye(4)
        initial_guess = last_pose @ prediction
        # print('initial guess: ', initial_guess)

        # Run ICP
        new_pose = register_frame(
            points=source,
            voxel_map=self.local_map,
            initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma,
            kernel=sigma / 3,
        )

        self.adaptive_threshold.update_model_deviation(np.linalg.inv(initial_guess) @ new_pose)
        self.local_map.update(frame_downsample, new_pose)
        self.poses.append(new_pose)
        return new_pose

    def voxelize(self, iframe):
        frame_downsample = voxel_down_sample(iframe, self.config.mapping.voxel_size * 0.5)
        source = voxel_down_sample(frame_downsample, self.config.mapping.voxel_size * 1.5)
        return source, frame_downsample

    def get_adaptive_threshold(self):
        return (
            self.config.adaptive_threshold.initial_threshold
            if not self.has_moved()
            else self.adaptive_threshold.get_threshold()
        )

    def get_prediction_model(self):
        if len(self.poses) < 2:
            return np.eye(4)
        return np.linalg.inv(self.poses[-2]) @ self.poses[-1]

    def has_moved(self):
        if len(self.poses) < 1:
            return False
        compute_motion = lambda T1, T2: np.linalg.norm((np.linalg.inv(T1) @ T2)[:3, -1])
        motion = compute_motion(self.poses[0], self.poses[-1])
        return motion > 5 * self.config.adaptive_threshold.min_motion_th

# ############################### ICP implementation ###########################################
# class egomotion:
#     def __init__(self, args):
#         self.args = args
#         self.thres = 0.25
#         self.init_pose = np.eye(4)
#         self.poses=[]

#     def remove_ground(self, pcd):
#         # cls = points[:,2] > self.args.min_z + self.args.ground_slack
#         # return points[cls]
#         # ground_idx, nonground_idx = segment_ground(points) # using patchwork++
#         # return points[nonground_idx]
#         # segment plane (ground)
#         _, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
#         pcd_rm_ground = pcd.select_by_index(inliers, invert=True)
#         return pcd_rm_ground

#     def register_frame(self, src, dst):
#         source = o3d.geometry.PointCloud()
#         source.points = o3d.utility.Vector3dVector(src.astype(np.float64))
#         target = o3d.geometry.PointCloud()
#         target.points = o3d.utility.Vector3dVector(dst.astype(np.float64))

#         source_rm_ground = self.remove_ground(source)
#         target_rm_ground = self.remove_ground(target)
#         # http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Fast-global-registration
#         reg_icp = o3d.pipelines.registration.registration_icp(
#         source_rm_ground, target_rm_ground, self.thres, self.init_pose,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000,
#                                                         relative_fitness=1e-6, 
#                                                         relative_rmse=1e-6,)
#                                                         )

#         pose = np.array(reg_icp.transformation)
#         self.poses.append(pose)
#         return pose
