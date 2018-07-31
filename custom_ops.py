# -*- coding: utf-8 -*-
# File: custom_ops.py
###
# Code are borrowed from tensorpack modified to support 5d input for batchnorm.
# https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/batch_norm.py
###

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages
import re
import six
import functools

from tensorpack.utils import logger
from tensorpack.utils.argtools import get_data_format
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.collection import backup_collection, restore_collection
from tensorpack import layer_register, VariableHolder
from tensorpack.tfutils.varreplace import custom_getter_scope

__all__ = ['BatchNorm', 'BatchRenorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4

def rename_get_variable(mapping):
    """
    Args:
        mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}
    Returns:
        A context where the variables are renamed.
    """
    def custom_getter(getter, name, *args, **kwargs):
        splits = name.split('/')
        basename = splits[-1]
        if basename in mapping:
            basename = mapping[basename]
            splits[-1] = basename
            name = '/'.join(splits)
        return getter(name, *args, **kwargs)
    return custom_getter_scope(custom_getter)

def map_common_tfargs(kwargs):
    df = kwargs.pop('data_format', None)
    if df is not None:
        df = get_data_format(df, tfmode=True)
        kwargs['data_format'] = df

    old_nl = kwargs.pop('nl', None)
    if old_nl is not None:
        kwargs['activation'] = lambda x, name=None: old_nl(x, name=name)

    if 'W_init' in kwargs:
        kwargs['kernel_initializer'] = kwargs.pop('W_init')

    if 'b_init' in kwargs:
        kwargs['bias_initializer'] = kwargs.pop('b_init')
    return kwargs

def convert_to_tflayer_args(args_names, name_mapping):
    """
    After applying this decorator:
    1. data_format becomes tf.layers style
    2. nl becomes activation
    3. initializers are renamed
    4. positional args are transformed to correspoding kwargs, according to args_names
    5. kwargs are mapped to tf.layers names if needed, by name_mapping
    """

    def decorator(func):
        @functools.wraps(func)
        def decorated_func(inputs, *args, **kwargs):
            kwargs = map_common_tfargs(kwargs)

            posarg_dic = {}
            assert len(args) <= len(args_names), \
                "Please use kwargs instead of positional args to call this model, " \
                "except for the following arguments: {}".format(', '.join(args_names))
            for pos_arg, name in zip(args, args_names):
                posarg_dic[name] = pos_arg

            ret = {}
            for name, arg in six.iteritems(kwargs):
                newname = name_mapping.get(name, None)
                if newname is not None:
                    assert newname not in kwargs, \
                        "Argument {} and {} conflicts!".format(name, newname)
                else:
                    newname = name
                ret[newname] = arg
            ret.update(posarg_dic)  # Let pos arg overwrite kw arg, for argscope to work

            return func(inputs, **ret)

        return decorated_func

    return decorator

def get_bn_variables(n_out, use_scale, use_bias, beta_init, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=beta_init)
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(1.0), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var,
                  moving_mean, moving_var, decay, internal_update):
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')

    if internal_update:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')
    else:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        return tf.identity(xn, name='output')


@layer_register()
@convert_to_tflayer_args(
    args_names=[],
    name_mapping={
        'use_bias': 'center',
        'use_scale': 'scale',
        'gamma_init': 'gamma_initializer',
        'decay': 'momentum',
        'use_local_stat': 'training'
    })
def BatchNorm3d(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_last',
              internal_update=False,
              sync_statistics=None):
    """
    Almost equivalent to `tf.layers.batch_normalization`, but different (and more powerful)
    in the following:
    1. Accepts an alternative `data_format` option when `axis` is None. For 2D input, this argument will be ignored.
    2. Default value for `momentum` and `epsilon` is different.
    3. Default value for `training` is automatically obtained from tensorpack's `TowerContext`, but can be overwritten.
    4. Support the `internal_update` option, which enables the use of BatchNorm layer inside conditionals.
    5. Support the `sync_statistics` option, which is very useful in small-batch models.
    Args:
        internal_update (bool): if False, add EMA update ops to
          `tf.GraphKeys.UPDATE_OPS`. If True, update EMA inside the layer by control dependencies.
          They are very similar in speed, but `internal_update=True` can be used
          when you have conditionals in your model, or when you have multiple networks to train.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/14699
        sync_statistics: either None or "nccl". By default (None), it uses statistics of the input tensor to normalize.
          When set to "nccl", this layer must be used under tensorpack multi-gpu trainers,
          and it then uses per-machine (multiple GPU) statistics to normalize.
          Note that this implementation averages the per-tower E[x] and E[x^2] among towers to compute
          global mean&variance. The result is the global mean&variance only if each tower has the same batch size.
          This option has no effect when not training.
          This option is also known as "Cross-GPU BatchNorm" as mentioned in https://arxiv.org/abs/1711.07240.
          Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/18222
    Variable Names:
    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.
    Note:
        Combinations of ``training`` and ``ctx.is_training``:
        * ``training == ctx.is_training``: standard BN, EMA are maintained during training
          and used during inference. This is the default.
        * ``training and not ctx.is_training``: still use batch statistics in inference.
        * ``not training and ctx.is_training``: use EMA to normalize in
          training. This is useful when you load a pre-trained BN and
          don't want to fine tune the EMA. EMA will not be updated in
          this case.
    """
    # parse shapes
    data_format = get_data_format(data_format, tfmode=False)
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    # in 3d conv, we have 5d dim [batch, c, d, h, w]
    # assert ndims in [2, 4], ndims
    if sync_statistics is not None:
        sync_statistics = sync_statistics.lower()
    assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

    if axis is None:
        if ndims == 2:
            data_format = 'NHWC'
            axis = 1
        elif ndims == 5:
            axis = 1 if data_format == 'NCHW' else 4
        else:
            axis = 1 if data_format == 'NCHW' else 3
    else:
        data_format = 'NCHW' if axis == 1 else 'NHWC'
    num_chan = shape[axis]

    # parse training/ctx
    ctx = get_current_tower_context()
    if training is None:
        training = ctx.is_training
    training = bool(training)
    TF_version = get_tf_version_number()
    if not training and ctx.is_training:
        assert TF_version >= 1.4, \
            "Fine tuning a BatchNorm model with fixed statistics is only " \
            "supported after https://github.com/tensorflow/tensorflow/pull/12580 "
        if ctx.is_main_training_tower:  # only warn in first tower
            logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
        # Using moving_mean/moving_variance in training, which means we
        # loaded a pre-trained BN and only fine-tuning the affine part.

    if sync_statistics is None or not (training and ctx.is_training):
        coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
        with rename_get_variable(
                {'moving_mean': 'mean/EMA',
                 'moving_variance': 'variance/EMA'}):
            tf_args = dict(
                axis=axis,
                momentum=momentum, epsilon=epsilon,
                center=center, scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                fused=True,
                _reuse=tf.get_variable_scope().reuse)
            if TF_version >= 1.5:
                tf_args['virtual_batch_size'] = virtual_batch_size
            else:
                assert virtual_batch_size is None, "Feature not supported in this version of TF!"
            layer = tf.layers.BatchNormalization(**tf_args)
            xn = layer.apply(inputs, training=training, scope=tf.get_variable_scope())

        # maintain EMA only on one GPU is OK, even in replicated mode.
        # because during training, EMA isn't used
        if ctx.is_main_training_tower:
            for v in layer.non_trainable_variables:
                add_model_variable(v)
        if not ctx.is_main_training_tower or internal_update:
            restore_collection(coll_bk)

        if training and internal_update:
            assert layer.updates
            with tf.control_dependencies(layer.updates):
                ret = tf.identity(xn, name='output')
        else:
            ret = tf.identity(xn, name='output')

        vh = ret.variables = VariableHolder(
            moving_mean=layer.moving_mean,
            mean=layer.moving_mean,  # for backward-compatibility
            moving_variance=layer.moving_variance,
            variance=layer.moving_variance)  # for backward-compatibility
        if scale:
            vh.gamma = layer.gamma
        if center:
            vh.beta = layer.beta
    else:
        red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])
        if ndims == 5:
            red_axis = [0, 2, 3, 4] if axis == 1 else [0, 1, 2, 3]
        new_shape = None  # don't need to reshape unless ...
        if ndims == 4 and axis == 1:
            new_shape = [1, num_chan, 1, 1]
        if ndims == 5 and axis == 1:
            new_shape = [1, num_chan, 1, 1, 1]

        batch_mean = tf.reduce_mean(inputs, axis=red_axis)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)

        if sync_statistics == 'nccl':
            if six.PY3 and TF_version <= 1.8 and ctx.is_main_training_tower:
                logger.warn("A TensorFlow bug will cause cross-GPU BatchNorm to fail. "
                            "Apply this patch: https://github.com/tensorflow/tensorflow/pull/20360")

            from tensorflow.contrib.nccl.ops import gen_nccl_ops
            shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
            num_dev = ctx.total
            batch_mean = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
            batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean_square,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
        elif sync_statistics == 'horovod':
            # Require https://github.com/uber/horovod/pull/331
            # Proof-of-concept, not ready yet.
            import horovod.tensorflow as hvd
            batch_mean = hvd.allreduce(batch_mean, average=True)
            batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
        batch_var = batch_mean_square - tf.square(batch_mean)
        batch_mean_vec = batch_mean
        batch_var_vec = batch_var

        beta, gamma, moving_mean, moving_var = get_bn_variables(
            num_chan, scale, center, beta_initializer, gamma_initializer)
        if new_shape is not None:
            batch_mean = tf.reshape(batch_mean, new_shape)
            batch_var = tf.reshape(batch_var, new_shape)
            # Using fused_batch_norm(is_training=False) is actually slightly faster,
            # but hopefully this call will be JITed in the future.
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                tf.reshape(beta, new_shape),
                tf.reshape(gamma, new_shape), epsilon)
        else:
            xn = tf.nn.batch_normalization(
                inputs, batch_mean, batch_var,
                beta, gamma, epsilon)

        if ctx.is_main_training_tower:
            ret = update_bn_ema(
                xn, batch_mean_vec, batch_var_vec, moving_mean, moving_var,
                momentum, internal_update)
        else:
            ret = tf.identity(xn, name='output')

        vh = ret.variables = VariableHolder(
            moving_mean=moving_mean,
            mean=moving_mean,  # for backward-compatibility
            moving_variance=moving_var,
            variance=moving_var)  # for backward-compatibility
        if scale:
            vh.gamma = gamma
        if center:
            vh.beta = beta
    return ret

