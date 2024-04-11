import os
import random
import argparse
import numpy as np
import open3d as o3d
import seaborn as sns
import scipy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from scipy.spatial.transform import Rotation as R

def plot_errors(args):
    with open(args.path) as f:
        lines = f.readlines()

    idxs_static = []
    EPE_static = []
    ACC3DS_static = []
    ACC3DR_static = []
    Outlier_static = []

    idxs_dynamic = []
    EPE_dynamic = []
    ACC3DS_dynamic = []
    ACC3DR_dynamic = []
    Outlier_dynamic = []

    for line in lines:

        if line.startswith('file: '):
            if ('overall' in line): continue
            print('line: ', line)
            chunks = line.split(', ')
            chunks = [chunk for chunk in chunks if ': ' in chunk]
            # print('chunks: ', chunks)
            assert len(chunks)==6
            for kk, chunk in enumerate(chunks):
                segs = chunk.split(': ')
                # print(f'{kk}- th chunk: {len(segs)}, {segs}')
                if kk==0: 
                    if 'static' in line:
                        idxs_static.append(segs[1]) 
                    else:
                        idxs_dynamic.append(segs[1]) 
                if kk==1: 
                    if 'static' in line:
                        EPE_static.append(float(segs[1])) 
                    else:
                        EPE_dynamic.append(float(segs[1])) 
                if kk==2: 
                    if 'static' in line:
                        ACC3DS_static.append(float(segs[1])) 
                    else: 
                        ACC3DS_dynamic.append(float(segs[1])) 
                if kk==3: 
                    if 'static' in line:
                        ACC3DR_static.append(float(segs[1])) 
                    else:
                        ACC3DR_dynamic.append(float(segs[1])) 
                if kk==4: 
                    if 'static' in line:
                        Outlier_static.append(float(segs[1])) 
                    else:
                        Outlier_dynamic.append(float(segs[1])) 

    assert len(EPE_static)==len(ACC3DS_static)
    assert len(EPE_static)==len(ACC3DR_static)
    assert len(EPE_static)==len(Outlier_static)
    assert len(EPE_static)==len(idxs_static)
    print(f'len(idxs): ', len(idxs_static))

    # fig = plt.figure()
    # ax1 = plt.subplot(121)
    # ax1.plot(np.arange(0, len(EPE_static)), np.array(EPE_static), c='g')
    # ax1.set_xticks(np.arange(0, len(idxs_static)), idxs_static)
    # ax1.set_ylim(0.0, 5)
    # ax1.set_title('static')
    # ax2 = plt.subplot(122)
    # ax2.plot(np.arange(0, len(EPE_dynamic)), np.array(EPE_dynamic), c='g')
    # ax2.set_xticks(np.arange(0, len(idxs_dynamic)), idxs_dynamic)
    # ax2.set_ylim(0.0, 5)
    # ax2.set_title('dynamic')
    # plt.suptitle(f'result over {len(EPE_static)}, {args.num_frames} per frame')
    # plt.show()
    # plt.close() 

    # for start_idx in np.arange(0, len(EPE_dynamic)-100, 100):
    #     EPE_dynamic_tmp = EPE_dynamic[start_idx:start_idx+100]
    #     idxs_dynamic_tmp = idxs_dynamic[start_idx:start_idx+100]
    #     fig = plt.figure()
    #     ax1 = plt.subplot(111)
    #     ax1.plot(np.arange(0, len(EPE_dynamic_tmp)), np.array(EPE_dynamic_tmp), c='g')
    #     ax1.set_xticks(np.arange(0, len(idxs_dynamic_tmp)), idxs_dynamic_tmp, rotation=90)
    #     # ax1.set_xlim(0, 100)
    #     ax1.set_ylim(0.0, 2.5)
    #     plt.suptitle(f'result over {start_idx}-{start_idx+100}: {len(EPE_dynamic_tmp)}, {args.num_frames} per frame')
    #     plt.show()
    #     plt.close() 


    return  { 
                "idxs_static"   : idxs_static,
                "EPE_static"    : EPE_static,
                "ACC3DS_static" : ACC3DS_static,
                "ACC3DR_static" : ACC3DR_static,
                "Outlier_static": Outlier_static,
                                             
                "idxs_dynamic"   : idxs_dynamic,
                "EPE_dynamic"    : EPE_dynamic,
                "ACC3DS_dynamic" : ACC3DS_dynamic,
                "ACC3DR_dynamic" : ACC3DR_dynamic,
                "Outlier_dynamic": Outlier_dynamic,
    }





if __name__ == "__main__":

    # Initialization
    random.seed(0)
    np.random.seed(0)
    # Parse hyperparameters
    parser = argparse.ArgumentParser(description='SceneFlow')
    parser.add_argument('--path', type=str, default='log_waymo.txt',
                        help='Path to dataset')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames per file (default: 5)')
    parser.add_argument('--if_verbose', default=False, action='store_true', 
                        help='visualize per segment')
    args = parser.parse_args()
    print(f'args: {args}')

    args.path = 'test_waymo_ego_icp.txt'
    result_cpu = plot_errors(args)
    args.path = 'test_waymo_ego_icp.txt'
    result_gpu = plot_errors(args)

    # fig = plt.figure()
    # ax1 = plt.subplot(221)
    # idxs_static = result_cpu['idxs_static']
    # ax1.plot(np.arange(0, len(idxs_static)), np.array(result_cpu['EPE_static']), c='g')
    # ax1.plot(np.arange(0, len(idxs_static)), np.array(result_gpu['EPE_static']), c='r', ls='-.')
    # ax1.set_xticks(np.arange(0, len(idxs_static)), idxs_static)
    # ax1.set_ylim(0.0, 5)
    # ax1.set_title('static')
    # ax2 = plt.subplot(222)

    # idxs_dynamic = result_cpu['idxs_dynamic']
    # ax2.plot(np.arange(0, len(idxs_dynamic)), np.array(result_cpu['EPE_dynamic']), c='g')
    # ax2.plot(np.arange(0, len(idxs_dynamic)), np.array(result_gpu['EPE_dynamic']), c='r', ls='-.')
    # ax2.set_xticks(np.arange(0, len(idxs_dynamic)), idxs_dynamic)
    # ax2.set_ylim(0.0, 5)
    # ax2.set_title('dynamic')

    # ax3 = plt.subplot(223)
    # dif = np.abs(np.array(result_cpu['EPE_static']) - np.array(result_gpu['EPE_static']))
    # ax3.plot(np.arange(0, len(idxs_static)), dif, c='b')
    # ax3.set_xticks(np.arange(0, len(idxs_static)), idxs_static)
    # ax3.set_ylim(0.0, 1)
    # ax3.set_title('static')

    # ax4 = plt.subplot(224)
    # dif = np.abs(np.array(result_cpu['EPE_dynamic']) - np.array(result_gpu['EPE_dynamic']))
    # ax4.plot(np.arange(0, len(idxs_dynamic)), dif, c='b')
    # ax4.set_xticks(np.arange(0, len(idxs_dynamic)), idxs_dynamic)
    # ax4.set_ylim(0.0, 1)
    # ax4.set_title('dynamic')

    # plt.suptitle(f'result over {len(idxs_static)}, {args.num_frames} per frame')
    # plt.show()
    # plt.close() 

    idxs_dynamic = result_cpu['idxs_dynamic']
    chunk = 100
    print('plot: ', len(idxs_dynamic), len(result_cpu['EPE_dynamic']), len(result_gpu['EPE_dynamic']))
    # assert len(idxs_dynamic)%chunk == 0
    for k in range(0, len(idxs_dynamic)//chunk):
        x1 = np.arange(0, chunk)
        y1 = np.array(result_cpu['EPE_dynamic'])[chunk*k : chunk*(k+1)]
        y2 = np.array(result_gpu['EPE_dynamic'])[chunk*k : chunk*(k+1)]
        anno = idxs_dynamic[chunk*k : chunk*(k+1)]
        print('plot: ', x1.shape, y1.shape, y2.shape)
        fig = plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(x1, y1, c='g')
        ax1.plot(x1, y2, c='r', ls='-.')
        ax1.set_xticks(x1, anno)
        ax1.set_ylim(0.0, 5)
        ax1.set_title('green vs red')

        plt.suptitle(f'result over {chunk*k}-{chunk*(k+1)}, {args.num_frames} per frame')
        plt.show()
        plt.close() 