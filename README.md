# ICP-Flow: LiDAR Scene Flow Estimation with ICP

Official implementation: [ICP-Flow: LiDAR Scene Flow Estimation with ICP](https://arxiv.org/abs/2402.17351) (CVPR 2024) 

[Yancong Lin](https://yanconglin.github.io/) and [Holger Caesar](https://sites.google.com/it-caesar.de/homepage/).

Delft University of Technology, The Netherlands.

Code release is in process, not ready to use yet!

## Introduction
Scene flow characterizes the 3D motion between two LiDAR scans captured by an autonomous vehicle at nearby timesteps. Prevalent methods consider scene flow as point-wise unconstrained flow vectors that can be learned by either large-scale training beforehand or time-consuming optimization at inference. However, these methods do not take into account that objects in autonomous driving often move rigidly. We incorporate this rigid-motion assumption into our design, where the goal is to associate objects over scans and then estimate the locally rigid transformations. We propose ICP-Flow, a learning-free flow estimator. The core of our design is the conventional Iterative Closest Point (ICP) algorithm, which aligns the objects over time and outputs the corresponding rigid transformations. Crucially, to aid ICP, we propose a histogram-based initialization that discovers the most likely translation, thus providing a good starting point for ICP. The complete scene flow is then recovered from the rigid transformations. We outperform state-of-the-art baselines, including supervised models, on the Waymo dataset and perform competitively on Argoverse-v2 and nuScenes. Further, we train a feedforward neural network, supervised by the pseudo labels from our model, and achieve top performance among all models capable of real-time inference. We validate the advantage of our model on scene flow estimation with longer temporal gaps, up to 0.4 seconds where other models fail to deliver meaningful results.

## Main Features: Motion Rigidity + Iterative Closesst Point

## Reproducing Results

### Installation

1. For easy reproducibility, you are suggested to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [anaconda](https://www.anaconda.com/distribution/) if you prefer) before following executing the following commands. 

```bash
conda create -f environment.yml
```
2. Install [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) for ground segmentation, by following [patchwork-plusplus/README.md](patchwork-plusplus/README.md). Once done, update the [path](https://github.com/yanconglin/ICP-Flow/blob/c93add6617a643e3c9db6b15c801b45e761411a5/utils_ground.py#L9) accordingly. It may take a while. There are several modifications on top of the original Patchwork++ to output point indices.

```bash
# To install Eigen
$ sudo apt-get install libeigen3-dev

# To install Open3D C++ packages
$ git clone https://github.com/isl-org/Open3D
$ cd Open3D
$ util/install_deps_ubuntu.sh # Only needed for Ubuntu
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install

# To install Patchwork++
$ cd patchwork-plusplus
$ mkdir build && cd build
$ cmake ..
$ make
```

### Dataset

1. Waymo and nuScenes: follow [Dynamic 3D Scene Analysis by Point Cloud Accumulation](https://github.com/prs-eth/PCAccumulation) to prepare the dataset, or you can simply run

```bash
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/PCAccumulation/data.zip
unzip data.zip
```

2. Argoverse-v2: see [ZeroFlow: Scalable Scene Flow via Distillation](https://github.com/kylevedder/zeroflow) for details.

   For simplicity, I took a different approach to load data: run ZeroFlow on the val/test set, and then save the points/labels in a .npz file per sample.
   See [dataset_argo.py](dataset_argo.py) for details.

   Alternatively, you may also load pretrained checkpoints (see below) to ZeroFlow and test on Argoverse-v2 using the ZeroFlow codebase. 
   

### Test
Configure all variables beforehand within the *.sh files. You may find detailed results at [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/AqrrbdMV6hnELpW). 

```bash
bash main.sh
```

### ICP-Flow + FNN

ICP-Flow + FNN has an identical design to ZeroFlow. You may download pre-trained weights from [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/AqrrbdMV6hnELpW) (todo). 

### Demo

```Bash
bash demo.sh
```

### Argoverse-v2 Scene Flow Challenge

Todo:  


### Citation
```bash
@article{lin2024icp,
  title={ICP-Flow: LiDAR Scene Flow Estimation with ICP},
  author={Lin, Yancong and Caesar, Holger},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
