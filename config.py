#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np
BASE_LR=0.01
BATCH_SIZE = 6
#PATCH_SIZE = [128, 128, 128]
PATCH_SIZE = [20, 144, 144]
INTENSITY_NORM = 'modality' # different norm method

BASEDIR = '/data/dataset/BRATS2018/'
TRAIN_DATASET = ['training']
VAL_DATASET = 'val'   # val or val17 
TEST_DATASET = 'test'
NUM_CLASS = 5

"""
python3 train.py --gpu 6 --load=./train_log/unet3d/model-50000 --evaluate eval
"""