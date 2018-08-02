# 3DUnet-Tensorflow
3D Unet biomedical segmentation model powered by tensorpack with fast io speed.

Borrow a lot of codes from https://github.com/taigw/brats17/. I improved the pipeline and using tensorpack's dataflow for faster io speed. Currently it takes around 7 minutes for 500 iterations with patch size [5 X 20 X 144 X 144]. You can achieve reasonable results within 40 epochs (more gpu will also reduce your training time.)

## Dependencies
+ Python 3; TensorFlow >= 1.4
+ Tensorpack^0.8.0 (https://github.com/tensorpack/tensorpack)
+ BRATS2017 or BRATS2018 data. It needs to have the following directory structure:
```
DIR/
  training/
    HGG/
    LGG/
  val/
    BRATS*.nii.gz
  test/
    BRATS*.nii.gz
```

## Usage
Change config in `config.py`:
1. Change `BASEDIR` to `/path/to/DIR` as described above.

Train:
```
python3 train.py --logdir=./train_log/unet3d --gpu 0
```
Eval:
```
python3 train.py --load=./train_log/unet3d/model-30000 --gpu 0 --eval
```
** If you want to use 5 fold cross validation:
1. Run generate_5fold.py to save 5fold.pkl
2. Set config CROSS_VALIDATION to True
3. Set config CROSS_VALIDATION_PATH to {/path/to/5fold.pkl}
4. Set config FOLD to {0~4}

## Results
The detailed parameters and training settings.
The results are derived from Brats2018 online evaluation on Validation Set.
### Setting 1:
Unet3d, num_filters=32 (all), depth=3
+ PatchSize = [5, 20, 144, 144] per gpu, num_gpus = 2, epochs = 40
+ Lr = 0.01, epoch time = 6:35(min), total_training_time ~ 5 hours
### Setting 2:
Unet3d, num_filters=32 (all), depth=3
+ PatchSize = [2, 128, 128, 128] pre gpu, num_gpus = 2, epochs = 40
+ Lr = 0.01, epoch time = 20:35(min), total_training_time ~ 10 hours
### Setting 3:
Unet3d, num_filters=16~256, **depth=5, **residual
+ PatchSize = [2, 128, 128, 128], num_gpus = 1, epochs = 40
+ Lr = 0.0005, epoch time = 6:35(min), total_training_time ~ 6 hours
### Setting 4:
Unet3d, num_filters=16~256, **depth=5, **residual, **deep-supervision, **InstanceNorm
+ PatchSize = [2, 128, 128, 128], num_gpus = 1, epochs = 40
+ Lr = 0.0005, epoch time = 6:35(min), total_training_time ~ 6 hours
### Setting 5:
Unet3d, num_filters=16~256, **depth=5, **residual, **deep-supervision, **InstanceNorm
+ PatchSize = [2, 128, 128, 128], num_gpus = 2, epochs = 40
+ Lr = 0.0005, epoch time = 6:35(min), total_training_time ~ 6 hours

| Setting | Dice_ET | Dice_WT | Dice_ET |
| --- | --- | --- | --- |
| 1 | 0.74 | 0.85 | 0.75 |
| 2 | 0.74 | 0.83 | 0.77 |
| 2* | 0.777 | 0.84 | 0.77 |

p.s. * means advanced post-processing
## Notes
Results for brats2018 will be updated and more experiments will be included. [2018/7/31]