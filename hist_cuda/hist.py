#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import glob
import math
import warnings
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, "cpp")
    tar_dir = os.path.join(src_dir, "build", ext_name)
    os.makedirs(tar_dir, exist_ok=True)
    srcs = glob.glob(f"{src_dir}/*.cu") + glob.glob(f"{src_dir}/*.cpp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.utils.cpp_extension import load

        ext = load(
            name=ext_name,
            sources=srcs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[],
            build_directory=tar_dir,
        )
    return ext


# defer calling load_cpp_ext to make CUDA_VISIBLE_DEVICES happy
global HIST
HIST = load_cpp_ext("HIST")

def hist(X, Y, min_x, min_y, min_z, max_x, max_y, max_z, len_x, len_y, len_z, mini_batch=8):
    # print('hist cuda params: ', X.shape, Y.shape,
    #       min_x, min_y, min_z,
    #       max_x, max_y, max_z,
    #       len_x, len_y, len_z,
    #       )
    histogram = HIST.hist(X, Y, 
                          min_x, min_y, min_z,
                          max_x, max_y, max_z, 
                          len_x, len_y, len_z, 
                          mini_batch
                          )
    return histogram

