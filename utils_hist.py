import numpy as np
import os
import sys
import open3d as o3d
import seaborn as sns
import scipy
import parmap
import warnings
import torchist
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_helper import nearest_neighbor_batch, random_choice
from utils_timer import MyTimer
from hist_cuda.hist import hist
import torch
warnings.filterwarnings('ignore')

# tok=1 already works decently. topk=5 is for ablation study and works slightly better
def topk_nms(x, k=5, kernel_size=11):
    b, h, w, d = x.shape
    x = x.unsqueeze(1)
    xp = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
    mask = (x == xp).float().clamp(min=0.0)
    xp = x * mask
    votes, idxs = torch.topk(xp.view(b, -1), dim=1, k=k)
    del xp, mask
    return votes, idxs.long()

def estimate_init_pose(args, src, dst):
    # visualize input
    # src_show = src.cpu().numpy()
    # dst_show = dst.cpu().numpy()
    # visualize_pcd(np.concatenate([src_show, dst_show], axis=0), 
    #                        np.concatenate([np.zeros((len(src_show)))+1, np.zeros((len(dst_show)))+2], axis=0), 
    #                        num_colors=3,
    #                        title=f'ransac initial: {len(src)} vs {len(dst)}'
    #                        )
    pcd1 = src[:, :, 0:3]
    pcd2 = dst[:, :, 0:3]
    mask1 = src[:, : , -1] > 0.0
    mask2 = dst[:, : , -1] > 0.0

    ###########################################################################################
    eps = 1e-8
    # https://pytorch.org/docs/stable/generated/torch.arange.html#torch-arange
    bins_x = torch.arange(-args.translation_frame, args.translation_frame+args.thres_dist-eps, args.thres_dist)
    bins_y = torch.arange(-args.translation_frame, args.translation_frame+args.thres_dist-eps, args.thres_dist)
    bins_z = torch.arange(-args.thres_dist, args.thres_dist+args.thres_dist-eps, args.thres_dist)
    # print(f'bins: {bins_x.min()} {bins_x.max()} {bins_x.shape}, {bins_z.min()} {bins_z.max()} {bins_z}')

    # bug there: when batch size is large!
    t_hist = hist(dst, src, 
                  bins_x.min(), bins_y.min(), bins_z.min(),
                  bins_x.max(), bins_y.max(), bins_z.max(),
                  len(bins_x), len(bins_y), len(bins_z))
    b, h, w, d = t_hist.shape
    # print(f't_hist.shape: {t_hist.shape} {bins_x.shape} {bins_y.shape} {bins_z.shape} {t_hist.max()}')
    ###########################################################################################

    t_maxs, t_argmaxs = topk_nms(t_hist)
    t_dynamic = torch.stack([ bins_x[t_argmaxs//d//w%h], bins_y[t_argmaxs//d%w], bins_z[t_argmaxs%d] ], dim=-1) + args.thres_dist//2
    # print(f't_dynamic: {t_dynamic.shape}, {t_maxs} {h}, {w}, {d}, {t_dynamic}')
    del t_hist, bins_x, bins_y, bins_z

    n = pcd1.shape[1]
    t_both = torch.cat([t_dynamic, t_dynamic.new_zeros(b, 1, 3)], dim=1)
    k = t_both.shape[1]

    pcd1_ = pcd1[:, None, :, :] + t_both[:, :, None, :]
    pcd2_ = pcd2[:, None, :, :].expand(-1, k, -1, -1)

    _, errors = nearest_neighbor_batch(pcd1_.reshape(b*k, n, 3), pcd2_.reshape(b*k, n, 3))
    _, errors_inv = nearest_neighbor_batch(pcd2_.reshape(b*k, n, 3), pcd1_.reshape(b*k, n, 3))

    # using inliers
    # inliers = torch.sum(torch.logical_and(errors < args.thres_dist*2, mask1[:, None, :]).float(), dim=-1)
    # inliers_inv = torch.sum(torch.logical_and(errors_inv < args.thres_dist*2, mask2[:, None, :]).float(), dim=-1)
    # inliers = torch.maximum(inliers, inliers_inv)
    # inlier, idx = inliers.max(dim=-1)
    # # print('inliers: ', inliers, idx)
    # error_best = inlier

    # using errors
    errors = (errors.view(b, k, n) * mask1[:, None, :]).sum(dim=-1) / mask1[:, None, :].sum(dim=-1)
    errors_inv = (errors_inv.view(b, k, n) * mask2[:, None, :]).sum(dim=-1) / mask2[:, None, :].sum(dim=-1)
    errors = torch.minimum(errors, errors_inv)
    error, idx = errors.min(dim=-1)
    error_best = error
    t_best = t_both[torch.arange(0, b, device=idx.device), idx, :]
    del pcd1_, pcd2_, errors
    # print(f'hist best: {t_best.shape}, {error_best.shape}')

    # # # # # # # # # # # to visualize
    # k = 0
    # print(f'hist best: {t_best[k]}, {error_best[k]}')
    # pcd1_best = (pcd1[k, mask1[k], 0:3]+t_best[k, None]).cpu().numpy()
    # pcd2_show = pcd2[k, mask2[k], 0:3].cpu().numpy()
    # visualize_pcd(np.concatenate([pcd1_best, pcd2_show], axis=0), 
    #             np.concatenate([np.zeros((len(pcd1_best)))+1, np.zeros((len(pcd2_show)))+2], axis=0), 
    #             num_colors=3,
    #             title=f'hist best: {sum(mask1[k])} vs {sum(mask2[k])}, error_best: {error_best[k]}'
    #             )

    transformation = torch.eye(4)[None].repeat(b, 1, 1)
    transformation[:, 0:3, -1] = t_best
    # print('hist transformation: ', transformation[k])
    return transformation
