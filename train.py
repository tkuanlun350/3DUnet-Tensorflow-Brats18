import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_number
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu
import config
from model import ( unet3d, Loss )
from data_sampler import (get_train_dataflow, get_eval_dataflow, get_test_dataflow)
from eval import (eval_brats, pred_brats, segment_one_image, segment_one_image_dynamic)

def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu


def get_model_output_names():
    ret = ['final_probs', 'final_pred']
    return ret


def get_model(modelType="training", inference_shape=config.INFERENCE_PATCH_SIZE):
    return Unet3dModel(modelType=modelType, inference_shape=inference_shape)

class Unet3dModel(ModelDesc):
    def __init__(self, modelType="training", inference_shape=config.INFERENCE_PATCH_SIZE):
        self.modelType = modelType
        self.inference_shape = inference_shape
        print(self.modelType)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=config.BASE_LR, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt
        
    def preprocess(self, image):
        # transform to NCHW
        return tf.transpose(image, [0, 4, 1, 2, 3])

    def inputs(self):
        S = config.PATCH_SIZE
        if self.modelType == 'training':
            ret = [
                tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 4), 'image'),
                tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'weight'),
                tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 1), 'label')]
        else:
            S = self.inference_shape
            ret = [
                tf.placeholder(tf.float32, (config.BATCH_SIZE, S[0], S[1], S[2], 4), 'image')]
        return ret

    def build_graph(self, *inputs):
        is_training = get_current_tower_context().is_training
        if is_training:
            image, weight, label = inputs
        else:
            image = inputs[0]
        image = self.preprocess(image)
        featuremap = unet3d('unet3d', image) # final upsampled feturemap
        if is_training:
            loss = Loss(featuremap, weight, label)
            wd_cost = regularize_cost(
                    '(?:unet3d)/.*kernel',
                    l2_regularizer(1e-5), name='wd_cost')

            total_cost = tf.add_n([loss, wd_cost], 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            final_probs = tf.nn.softmax(featuremap, name="final_probs") #[b,d,h,w,num_class]
            final_pred = tf.argmax(final_probs, axis=-1, name="final_pred")

class EvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'], get_model_output_names())
        self.df = get_eval_dataflow()
    
    def _eval(self):
        scores = eval_brats(self.df, lambda img: segment_one_image(img, [self.pred], is_online=True))
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num > 0 and self.epoch_num % config.EVAL_EPOCH == 0:
            self._eval()

def offline_evaluate(pred_func, output_file):
        df = get_eval_dataflow()
        if config.DYNAMIC_SHAPE_PRED:    
            eval_brats(
                df, lambda img: segment_one_image_dynamic(img, pred_func))
        else:
            eval_brats(
                df, lambda img: segment_one_image(img, pred_func))

def offline_pred(pred_func, output_file):
        df = get_test_dataflow()
        pred_brats(
            df, lambda img: segment_one_image(img, pred_func))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to all availalbe ones')
    parser.add_argument('--load', help='load model for evaluation or training')
    parser.add_argument('--logdir', help='log directory', default='train_log/unet3d')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', action='store_true', help="Run evaluation")
    parser.add_argument('--predict', action='store_true', help="Run prediction")
    args = parser.parse_args()
    
    if args.datadir:
        config.BASEDIR = args.datadir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate:
        if config.DYNAMIC_SHAPE_PRED:
            def get_dynamic_pred(shape):
                return OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference", inference_shape=shape),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            offline_evaluate([get_dynamic_pred], args.evaluate)
        elif config.MULTI_VIEW:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred1 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_sa/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred2 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_cr/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            offline_evaluate([pred, pred1, pred2], args.evaluate)
        else:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            # autotune is too slow for inference
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            assert args.load
            offline_evaluate([pred], args.evaluate)
    
    elif args.predict:
        if config.MULTI_VIEW:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred1 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_sa/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            pred2 = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader("./train_log/unet3d_8_N4_cr/model-10000"),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            # autotune is too slow for inference
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            offline_pred([pred, pred1, pred2], args.evaluate)
        else:
            pred = OfflinePredictor(PredictConfig(
                    model=get_model(modelType="inference"),
                    session_init=get_model_loader(args.load),
                    input_names=['image'],
                    output_names=get_model_output_names()))
            # autotune is too slow for inference
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            assert args.load
            offline_pred([pred], args.evaluate)
        
    else:
        logger.set_logger_dir(args.logdir)
        factor = get_batch_factor()
        stepnum = config.STEP_PER_EPOCH
        
        cfg = TrainConfig(
            model=get_model(),
            data=QueueInput(get_train_dataflow()),
            callbacks=[
                PeriodicCallback(
                    ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                    every_k_epochs=20),
                ScheduledHyperParamSetter('learning_rate', 
                    [(40, config.BASE_LR*0.1),
                    (60, config.BASE_LR*0.01)]
                ),
                #EvalCallback(),
                GPUUtilizationTracker(),
                PeakMemoryTracker(),
                EstimatedTimeLeft(),
            ],
            steps_per_epoch=stepnum,
            max_epoch=80,
            session_init=get_model_loader(args.load) if args.load else None,
        )
        # nccl mode gives the best speed
        trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu(), mode='nccl')
        launch_train_with_config(cfg, trainer)
        