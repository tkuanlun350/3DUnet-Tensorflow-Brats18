#####
# some functions are borrowed from https://github.com/taigw/brats17/
#####

import cv2
import numpy as np
import copy
import os, sys
from tqdm import tqdm
from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    BatchData, MapData, imgaug, TestDataSpeed, MultiProcessMapData,
    MapDataComponent, DataFromList, PrefetchDataZMQ)
import tensorpack.utils.viz as tpviz
"""
from utils.np_box_ops import iou as np_iou
from utils.np_box_ops import area as np_area
from utils.generate_anchors import generate_anchors
from utils.box_ops import get_iou_callable
"""
from  data_loader import BRATS_SEG
import config
from tensorpack.dataflow import RNGDataFlow
import nibabel
import SimpleITK as sitk
import random
import time
from utils import *

class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def size(self):
        return self._size

    def get_data(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


def sampler3d(im, gt, with_gt=True):
    """
    sample 3d volume to size (depth, h, w, 4)
    """
    mods = sorted(im.keys())
    volume_list = []
    for mod_idx, mod in enumerate(mods):
        filename = im[mod]
        volume = load_nifty_volume_as_array(filename, with_header=False)
        # 155 244 244
        if mod_idx == 0:
            # contain whole tumor
            margin = 5 # small padding value
            bbmin, bbmax = get_none_zero_region(volume, margin)
        #s1 = time.time()
        volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
        #print("crop", time.time()-s1)
        if mod_idx == 0:
            weight = np.asarray(volume > 0, np.float32)
        #s2 = time.time()
        if config.INTENSITY_NORM == 'modality':
            volume = itensity_normalize_one_volume(volume)
        #print("norm", time.time()-s2)
        volume_list.append(volume)
    ## volume_list [(depth, h, w)*4]
    if with_gt:
        label = load_nifty_volume_as_array(gt, False)
        label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)
    data_shape = config.PATCH_SIZE
    label_shape = config.PATCH_SIZE
    data_slice_number = data_shape[0]
    label_slice_number = label_shape[0]
    volume_shape = volume_list[0].shape
    sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
    sub_label_shape =[label_slice_number, label_shape[1], label_shape[2]]
    flag = False
    while flag == False:
        center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, "full", None)
        sub_label = extract_roi_from_volume(label,
                                            center_point,
                                            sub_label_shape,
                                            fill = 'zero')
        if sub_label.sum() > 0:
            flag = True
    sub_data = []
    flip = random.random() > 0.5
    for moda in range(len(volume_list)):
        sub_data_moda = extract_roi_from_volume(volume_list[moda],center_point,sub_data_shape)
        if(flip):
            sub_data_moda = np.flip(sub_data_moda, -1)
        sub_data.append(sub_data_moda)
    sub_data = np.array(sub_data) #4, depth, h, w
    sub_weight = extract_roi_from_volume(weight,
                                            center_point,
                                            sub_label_shape,
                                            fill = 'zero')
    if(flip):
        sub_weight = np.flip(sub_weight, -1)
    #sub_label = extract_roi_from_volume(label,
    #                                        center_point,
    #                                        sub_label_shape,
    #                                        fill = 'zero')
    if(flip):
        sub_label = np.flip(sub_label, -1)
    batch = {}
    axis = [1,2,3,0] #[1,2,3,0] [d, h, w, modalities]
    batch['images']  = np.transpose(sub_data, axis)
    batch['weights'] = np.transpose(sub_weight[np.newaxis, ...], axis)
    batch['labels']  = np.transpose(sub_label[np.newaxis, ...], axis)
    # other meta info ?
    return batch

def sampler3d_whole(im):
    mods = sorted(im.keys())
    volume_list = []
    for mod_idx, mod in enumerate(mods):
        filename = im[mod]
        volume = load_nifty_volume_as_array(filename, with_header=False)
        if config.INTENSITY_NORM == 'modality':
            volume = itensity_normalize_one_volume(volume)        
        volume_list.append(volume)
    sub_data = np.array(volume_list)
    batch = {}
    axis = [1,2,3,0] #[1,2,3,0] [d, h, w, modalities]
    batch['images']  = np.transpose(sub_data, axis)
    
    return batch

def get_train_dataflow(add_mask=True):
    """
    
    """
    imgs = BRATS_SEG.load_many(
        config.BASEDIR, config.TRAIN_DATASET, add_gt=False, add_mask=add_mask)
    # no filter for training
    imgs = list(filter(lambda img: len(img['gt']) > 0, imgs))    # log invalid training

    ds = DataFromList(imgs, shuffle=True)
    
    def preprocess(data):
        fname, gt, im = data['file_name'], data['gt'], data['image_data']
        assert im is not None, fname
        flag = False
        batch = sampler3d(im, gt)
        assert batch['labels'].sum() > 0
        return [batch['images'], batch['weights'], batch['labels']]
        
    ds = BatchData(MapData(ds, preprocess), config.BATCH_SIZE)
    ds = PrefetchDataZMQ(ds, 6)
    return ds

def get_eval_dataflow():
    imgs = BRATS_SEG.load_many(config.BASEDIR, config.VAL_DATASET, add_gt=False)
    # no filter for training
    ds = DataFromListOfDict(imgs, ['file_name', 'id', 'image_data'])

    def f(data):
        batch = sampler3d_whole(data)
        return batch
    ds = MapDataComponent(ds, f, 2)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

if __name__ == "__main__":
    df = get_train_dataflow()
    df.reset_state()
    for i in tqdm(df.get_data()):
        pass
        """
        for k in i:
            arr = i[k][0]
            arr = np.rot90(arr, k=2, axes= (1,2))
            OUTPUT_AFFINE = np.array(
                    [[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])
            if k == 'labels':
                img = nibabel.Nifti1Image(arr.astype(np.int32), OUTPUT_AFFINE)
            else:
                img = nibabel.Nifti1Image(arr.astype(np.float32), OUTPUT_AFFINE)
            #nibabel.save(img, "./debug_output/{}.nii".format(k))
        break
        """
