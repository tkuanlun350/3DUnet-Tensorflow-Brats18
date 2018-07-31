# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
import numpy as np
import cv2

from tensorpack.utils.utils import get_tqdm_kwargs
import config
from utils import *

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])

def segment_one_image(data, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """
    img = data['images']
    img_shape = img.shape
    depth_step = config.PATCH_SIZE[0]
    results = np.array([])
    results_prob = np.array([])
    # pad z to factor 20
    _d = (img_shape[0] // depth_step + 1) * depth_step - img_shape[0] # pad depth % depth_step
    img = np.pad(img, ((0,_d),(0,0),(0,0),(0,0)), "constant")
    img = img[np.newaxis, ...] # add batch dim

    for depth in range(0, img.shape[1], depth_step):
        im = img[:, depth: depth + depth_step, :, :, :]
        final_probs, final_pred = model_func(im)
        if results.shape[0] == 0:
            results = final_pred[0]
            results_prob = final_probs[0]
        else:
            results = np.concatenate((results, final_pred[0]), axis=0)
            results_prob = np.concatenate((results_prob, final_probs[0]), axis=0)
    results = results[0:img_shape[0], :, :]
    results_prob = results_prob[0:img_shape[0], :, :]
    return results, results_prob

def dice_of_brats_data_set(gt, pred, type_idx):
    dice_all_data = []
    for i in range(len(gt)):
        g_volume = gt[i]
        s_volume = pred[i]
        dice_one_volume = []
        if(type_idx ==0): # whole tumor
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        elif(type_idx == 1): # tumor core
            s_volume[s_volume == 2] = 0
            g_volume[g_volume == 2] = 0
            temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            dice_one_volume = [temp_dice]
        else:
            for label in [1, 2, 3, 4]: # dice of each class
                temp_dice = binary_dice3d(s_volume == label, g_volume == label)
                dice_one_volume.append(temp_dice)
        dice_all_data.append(dice_one_volume)
    return dice_all_data

def eval_brats(df, detect_func, with_gt=True):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    gts = []
    results = []
    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for filename, image_id, data in df.get_data():
            pred, probs = detect_func(data)
            if with_gt == False:
                # just save to folder
                save_to_nii(pred, image_id, outdir="eval_out18", mode="label")
                # save prob to ensemble
                # save_to_pkl(probs, image_id, outdir="eval_out18_prob_{}".format(config.CROSS_VALIDATION))
                pbar.update()
                continue
            gt = load_nifty_volume_as_array("{}/{}_seg.nii.gz".format(filename, image_id))
            gts.append(gt)
            results.append(pred)
            pbar.update()
    if with_gt:
        test_types = ['whole','core', 'all']
        ret = {}
        for type_idx in range(3):
            dice = dice_of_brats_data_set(gts, results, type_idx)
            dice = np.asarray(dice)
            dice_mean = dice.mean(axis = 0)
            dice_std  = dice.std(axis  = 0)
            test_type = test_types[type_idx]
            ret[test_type] = dice_mean
            print('tissue type', test_type)
            if(test_type == 'all'):
                print('tissue label', [1, 2, 3, 4])
            print('dice mean  ', dice_mean)
            # print('dice std   ', dice_std)
    return ret
 
