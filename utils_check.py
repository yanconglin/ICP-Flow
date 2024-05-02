import torch
import numpy as np
import open3d as o3d
import seaborn as sns
import pytorch3d.transforms as pytorch3d_t
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
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_helper import get_bbox_tensor
warnings.filterwarnings('ignore')

def sanity_check(args, src_points, dst_points, src_labels, dst_labels, pairs):
    pairs_true = []
    for pair in pairs:
        src = src_points[src_labels==pair[0]]
        dst = dst_points[dst_labels==pair[1]]
        # print('sanity check :', pair, len(src), len(dst), src.mean(0), dst.mean(0), torch.linalg.norm(dst.mean(0) - src.mean(0)), args.translation_frame)

        # scenario 1: either src or dst does not exist, return None
        # scenario 2: both src or dst exist, but they are not matchable because of ground points/too few points/size mismatch, return False
        # scenario 3: both src or dst exist, and they are are matchable, return True
        if min(len(src), len(dst))<args.min_cluster_size: continue
        if min(pair[0], pair[1])<0: continue  # ground or non-clustered points

        mean_src = src.mean(0)
        mean_dst = dst.mean(0)
        if torch.linalg.norm((mean_dst - mean_src)[0:2])>args.translation_frame: continue # x/y translation

        src_bbox = get_bbox_tensor(src)
        dst_bbox = get_bbox_tensor(dst)
        # print('sanity check bbox:', src_bbox, dst_bbox)
        if min(src_bbox[0], dst_bbox[0]) < args.thres_box * max(src_bbox[0], dst_bbox[0]): continue 
        if min(src_bbox[1], dst_bbox[1]) < args.thres_box * max(src_bbox[1], dst_bbox[1]): continue 
        if min(src_bbox[2], dst_bbox[2]) < args.thres_box * max(src_bbox[2], dst_bbox[2]): continue 

        pairs_true.append(pair)
    if len(pairs_true)>0:
        return torch.vstack(pairs_true)
    else:
        return torch.zeros((0,2))

def check_transformation(args, translation, rotation, iou):
    # print('check transformation: ', translation, rotation, iou, args.translation_frame)
    # # # check translation
    if torch.linalg.norm(translation) > args.translation_frame:
        return False

    # # # check iou
    if iou<args.thres_iou:
        return  False

    # # # check rotation, in degrees, almost no impact on final result
    max_rot = args.thres_rot * 90.0
    if torch.abs(rotation[1:3]).max()>max_rot: # roll and pitch
        return False

    return True
