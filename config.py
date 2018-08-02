#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# unet model
RESIDUAL = True
DEPTH = 5
DEEP_SUPERVISION = False
FILTER_GROW = True
INSTANCE_NORM = False
# Use multi-view fusion 3 models for 3 view must be trained
DIRECTION = 'axial' # axial, sagittal, coronal
MULTI_VIEW = False

# training config
BASE_LR = 0.001
###
# Use when 5 fold cross validation
# 1. First run generate_5fold.py to save 5fold.pkl
# 2. Set CROSS_VALIDATION to True
# 3. CROSS_VALIDATION_PATH to /path/to/5fold.pkl
# 4. Set FOLD to {0~4}
###
CROSS_VALIDATION = False
CROSS_VALIDATION_PATH = "./5fold.pkl"
FOLD = 0
# change PATCH_SIZE in inference if cropped brain region > PATCH_SIZE
DYNAMIC_SHAPE_PRED = False 
ADVANCE_POSTPROCESSING = True
BATCH_SIZE = 2
PATCH_SIZE = [128, 128, 128]
INTENSITY_NORM = 'modality' # different norm method
STEP_PER_EPOCH = 500
EVAL_EPOCH = 10

# data path
BASEDIR = '/data/dataset/BRATS2018/'
TRAIN_DATASET = ['training']
VAL_DATASET = 'val'   # val or val17 
TEST_DATASET = 'val'
NUM_CLASS = 4
