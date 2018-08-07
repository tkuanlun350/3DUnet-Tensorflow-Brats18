#####
# some functions are borrowed from https://github.com/taigw/brats17/
#####

import cv2
import numpy as np
import copy
import os, sys
from tqdm import tqdm
from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (ProxyDataFlow,
    MapData, TestDataSpeed, MultiProcessMapData,
    MapDataComponent, DataFromList, PrefetchDataZMQ)
from  data_loader import BRATS_SEG
import config
from tensorpack.dataflow import RNGDataFlow
import nibabel
import random
import time
from utils import *
import six

class BatchData(ProxyDataFlow):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `ds.size()` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = int(batch_size)
        self.remainder = remainder
        self.use_list = use_list

    def size(self):
        ds_size = self.ds.size()
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            if config.DATA_SAMPLING == 'all_positive':
                if data[2].sum() == 0:
                    continue
            elif config.DATA_SAMPLING == 'one_positive':
                if len(holder) == self.batch_size - 1:
                    # force to contain label
                    t = [x[2].sum() for x in holder]
                    if sum(t) == 0 and data[2].sum() == 0:
                        continue
            else:
                pass
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchData._aggregate_batch(holder, self.use_list)

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                if type(dt) in list(six.integer_types) + [bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    try:
                        tp = dt.dtype
                    except AttributeError:
                        raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                try:
                    result.append(
                        np.asarray([x[k] for x in data_holder], dtype=tp))
                except Exception as e:  # noqa
                    logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    if isinstance(dt, np.ndarray):
                        s = pprint.pformat([x[k].shape for x in data_holder])
                        logger.error("Shape of all arrays to be batched: " + s)
                    try:
                        # open an ipython shell if possible
                        import IPython as IP; IP.embed()    # noqa
                    except ImportError:
                        pass
        return result

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

def sampler3d(volume_list, label, weight):
    """
    sample 3d volume to size (depth, h, w, 4)
    """
    margin = 5
    volume_list = transpose_volumes(volume_list, config.DIRECTION)
    data_shape = config.PATCH_SIZE
    label_shape = config.PATCH_SIZE
    data_slice_number = data_shape[0]
    label_slice_number = label_shape[0]
    volume_shape = volume_list[0].shape
    sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
    sub_label_shape =[label_slice_number, label_shape[1], label_shape[2]]

    label = transpose_volumes(label, config.DIRECTION)
    center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, "full", None)
    sub_label = extract_roi_from_volume(label,
                                        center_point,
                                        sub_label_shape,
                                            fill = 'zero')
    sub_data = []
    flip = False
    for moda in range(len(volume_list)):
        sub_data_moda = extract_roi_from_volume(volume_list[moda],center_point,sub_data_shape)
        if(flip):
            sub_data_moda = np.flip(sub_data_moda, -1)
        sub_data.append(sub_data_moda)
    sub_data = np.array(sub_data) #4, depth, h, w

    weight = transpose_volumes(weight, config.DIRECTION)
    sub_weight = extract_roi_from_volume(weight,
                                            center_point,
                                            sub_label_shape,
                                            fill = 'zero')
    if(flip):
        sub_weight = np.flip(sub_weight, -1)
    
    if(flip):
        sub_label = np.flip(sub_label, -1)
    batch = {}
    axis = [1,2,3,0] #[1,2,3,0] [d, h, w, modalities]
    batch['images']  = np.transpose(sub_data, axis)
    batch['weights'] = np.transpose(sub_weight[np.newaxis, ...], axis)
    batch['labels']  = np.transpose(sub_label[np.newaxis, ...], axis)
    # other meta info ?
    return batch

def sampler3d_whole(volume_list, label, weight, original_shape, bbox):
    """
    mods = sorted(im.keys())
    volume_list = []
    for mod_idx, mod in enumerate(mods):
        filename = im[mod]
        volume = load_nifty_volume_as_array(filename, with_header=False)
        if mod_idx == 0:
            # contain whole tumor
            margin = 5 # small padding value
            original_shape = volume.shape
            bbmin, bbmax = get_none_zero_region(volume, margin)
        volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
        if mod_idx == 0:
            weight = np.asarray(volume > 0, np.float32)
        if config.INTENSITY_NORM == 'modality':
            volume = itensity_normalize_one_volume(volume)        
        volume_list.append(volume)
    """
    sub_data = np.array(volume_list)
    batch = {}
    axis = [1,2,3,0] #[1,2,3,0] [d, h, w, modalities]
    batch['images']  = np.transpose(sub_data, axis)
    batch['weights'] = np.transpose(weight[np.newaxis, ...], axis)
    batch['original_shape'] = original_shape
    batch['bbox'] = bbox
    
    return batch

def get_train_dataflow(add_mask=True):
    """
    
    """
    if config.CROSS_VALIDATION:
        imgs = BRATS_SEG.load_from_file(config.BASEDIR, config.TRAIN_DATASET)
    else:
        imgs = BRATS_SEG.load_many(
            config.BASEDIR, config.TRAIN_DATASET, add_gt=False, add_mask=add_mask)
    # no filter for training
    imgs = list(imgs) 

    ds = DataFromList(imgs, shuffle=True)
    
    def preprocess(data):
        if config.NO_CACHE:
            fname, gt, im = data['file_name'], data['gt'], data['image_data']
            volume_list, label, weight, _, _ = crop_brain_region(im, gt)
            batch = sampler3d(volume_list, label, weight)
        else:
            volume_list, label, weight, _, _ = data['preprocessed']
            batch = sampler3d(volume_list, label, weight)
        return [batch['images'], batch['weights'], batch['labels']]
        
    ds = BatchData(MapData(ds, preprocess), config.BATCH_SIZE)
    ds = PrefetchDataZMQ(ds, 6)
    return ds

def get_eval_dataflow():
    #if config.CROSS_VALIDATION:
    imgs = BRATS_SEG.load_from_file(config.BASEDIR, config.VAL_DATASET)
    # no filter for training
    ds = DataFromListOfDict(imgs, ['file_name', 'id', 'preprocessed'])

    def f(data):
        volume_list, label, weight, original_shape, bbox = data
        batch = sampler3d_whole(volume_list, label, weight, original_shape, bbox)
        return batch
    ds = MapDataComponent(ds, f, 2)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

def get_test_dataflow():
    imgs = BRATS_SEG.load_many(config.BASEDIR, config.TEST_DATASET, add_gt=False)
    # no filter for training
    ds = DataFromListOfDict(imgs, ['file_name', 'id', 'preprocessed'])

    def f(data):
        volume_list, label, weight, original_shape, bbox = data
        batch = sampler3d_whole(volume_list, label, weight, original_shape, bbox)
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
