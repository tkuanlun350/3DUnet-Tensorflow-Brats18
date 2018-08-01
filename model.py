###
# Loss functions are modified from NiftyNet
###

import tensorflow as tf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope

from tensorpack.models import (
    BatchNorm, layer_register
)
from custom_ops import BatchNorm3d
import numpy as np
import config
import tensorflow.contrib.slim as slim
PADDING = "SAME"
DATA_FORMAT="channels_first"

@layer_register(log_shape=True)
def unet3d(inputs):
    print("inputs", inputs.shape)
    depth = 3
    down_list = []
    layer = inputs
    
    for d in range(depth):
        layer = Unet3dBlock('down{}'.format(d), layer, kernels=(3,3,3), n_feat=32, s=1)
        down_list.append(layer)
        if d != depth - 1:
            layer = tf.layers.max_pooling3d(inputs=layer, 
                                            pool_size=(2,2,2), 
                                            strides=2, 
                                            padding=PADDING, 
                                            data_format=DATA_FORMAT,
                                            name="pool_{}".format(d))
        print("layer", layer.shape)
    for d in range(depth-1):
        layer = tf.layers.conv3d_transpose(inputs=layer, 
                                    filters=32,
                                    kernel_size=(2,2,2),
                                    strides=2,
                                    padding=PADDING,
                                    activation=tf.nn.relu,
                                    data_format=DATA_FORMAT,
                                    name="up_conv_{}".format(d))
        if DATA_FORMAT == 'channels_first':
            layer = tf.concat([layer, down_list[depth-1-1-d]], axis=1)
        else:
            layer = tf.concat([layer, down_list[depth-1-1-d]], axis=-1)
        print("concat", layer.shape)
        layer = Unet3dBlock('up{}'.format(d), layer, kernels=(3,3,3), n_feat=32, s=1)
        print("layer", layer.shape)
    layer = tf.layers.conv3d(layer, 
                            filters=config.NUM_CLASS,
                            kernel_size=(1,1,1),
                            padding="SAME",
                            activation=tf.identity,
                            data_format=DATA_FORMAT,
                            name="final")
    if DATA_FORMAT == 'channels_first':
        layer = tf.transpose(layer, [0, 2, 3, 4, 1]) # to-channel last
    print("final", layer.shape) # [3, num_class, d, h, w]
    return layer

def BN_Relu(x):
    l = BatchNorm3d('bn', x, axis=1 if DATA_FORMAT == 'channels_first' else -1)
    l = tf.nn.relu(l)
    return l

def Unet3dBlock(prefix, l, kernels, n_feat, s):
    for i in range(2):
        l = tf.layers.conv3d(inputs=l, 
                   filters=n_feat,
                   kernel_size=kernels,
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_Relu(x),
                   data_format=DATA_FORMAT,
                   name="{}_conv_{}".format(prefix, i))
    return l
### from niftynet ####
def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
        tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    return 1 - generalised_dice_score


def dice(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in
        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016
    using a square in the denominator
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))
    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    
    return 1.0 - tf.reduce_mean(dice_score)

def Loss(feature, weight, gt):
    # compute batch-wise
    losses = []
    for idx in range(config.BATCH_SIZE):
        f = tf.reshape(feature[idx], [-1, config.NUM_CLASS])
        #f = tf.cast(f, dtype=tf.float32)
        #f = tf.nn.softmax(f)
        w = tf.reshape(weight[idx], [-1])
        g = tf.reshape(gt[idx], [-1])
        print(f.shape, w.shape, g.shape)
        if g.shape.as_list()[-1] == 1:
            g = tf.squeeze(g, axis=-1) # (nvoxel, )
        if w.shape.as_list()[-1] == 1:
            w = tf.squeeze(w, axis=-1) # (nvoxel, )
        f = tf.nn.softmax(f)
        loss_per_batch = generalised_dice_loss(f, g, weight_map=w)
        #loss_per_batch = cross_entropy(f, g, weight_map=w)
        losses.append(loss_per_batch)
    return tf.reduce_mean(losses, name="dice_loss")

    
if __name__ == "__main__":
    image = tf.transpose(tf.constant(np.zeros((config.BATCH_SIZE,20,144,144,4)).astype(np.float32)), [0,4,1,2,3])
    gt = tf.constant(np.zeros((config.BATCH_SIZE,20,144,144,1)).astype(np.float32))
    weight = tf.constant(np.ones((config.BATCH_SIZE,20,144,144,1)).astype(np.float32))
    t = unet3d('unet3d', image)
    loss = Loss(t, weight, gt)
    print(t.shape, loss)