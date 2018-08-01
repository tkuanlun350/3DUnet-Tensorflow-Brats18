#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np
# training config
BASE_LR=0.01
# change PATCH_SIZE in inference if cropped brain region > PATCH_SIZE
DYNAMIC_SHAPE_PRED = False 
MULTI_VIEW = False
BATCH_SIZE = 6
PATCH_SIZE = [20, 144, 144]
INTENSITY_NORM = 'modality' # different norm method
STEP_PER_EPOCH = 500
EVAL_EPOCH = 2

# data path
BASEDIR = '/data/dataset/BRATS2018/'
TRAIN_DATASET = ['training']
VAL_DATASET = 'val'   # val or val17 
TEST_DATASET = 'val'
NUM_CLASS = 5

"""
python3 train.py --gpu 6 --load=./train_log/unet3d/model-50000 --evaluate eval
"""