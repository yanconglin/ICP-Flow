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
import torch
import open3d as o3d
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_match import match_pcds
from utils_helper import transform_points
from utils_cluster import cluster_pcd
from utils_ground import segment_ground, segment_ground_thres, segment_ground_pypatchworkpp
from utils_ego_motion import egomotion
from timeit import default_timer as timer
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)

def track(args, point_src, point_dst, label_src, label_dst):
    pairs, transformations = match_pcds(args, point_src, point_dst, label_src, label_dst) 
    # print(f'match_pcds pairs: {pairs}, {transformations}')
    if args.if_verbose: print(f'match_pcds pairs: {torch.round(pairs[:, 0:2], decimals=2)}')
    return pairs, transformations

    # # # # # # # # # # # # # # # # print(f'##################### DEBUG ###############################')
    # # # # # # # # # # # # # # # # # # Debug near static
    # label_src_debug = np.zeros((len(point_src)))-1e8
    # label_dst_debug = np.zeros((len(point_dst)))-1e8
    # label_src_debug[label_src==1]= 1
    # label_dst_debug[label_dst==1]= 1
    # visualize_pcd(np.concatenate([point_src, point_dst], axis=0), np.concatenate([label_src_debug, label_dst_debug+1], axis=0), 
    #               num_colors=3, title=f'debug')
    # pairs, transformations = match_pcds(args, point_src, point_dst, label_src_debug, label_dst_debug) 
    # print(f'match_pcds pairs: {pairs}, {transformations}')

    # # # # # # # # # # # # # # # # # # Debug dynamic
    # label_src_debug = np.zeros((len(point_src)))-1e8
    # label_dst_debug = np.zeros((len(point_dst)))-1e8
    # label_src_debug[label_src==30]=1
    # label_dst_debug[label_dst==29]=2
    # # label_dst_debug[label_dst==75]=3
    # visualize_pcd(np.concatenate([point_src, point_dst], axis=0), np.concatenate([label_src_debug, label_dst_debug], axis=0), 
    #               num_colors=10, title=f'debug')
    # pairs, transformations = match_pcds(args, point_src, point_dst, label_src_debug, label_dst_debug) 
    # print(f'match_pcds pairs: {pairs}, {transformations}')
    # # # # # # # # # # # # # # # # print(f'##################### DEBUG ###############################')
