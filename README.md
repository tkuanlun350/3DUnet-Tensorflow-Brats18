# 3DUnet-Tensorflow
3D Unet biomedical segmentation model powered by tensorpack with fast io speed.

Borrow a lot of codes from https://github.com/taigw/brats17/. I improved the pipeline and using tensorpack's dataflow for faster io speed. Currently it takes 7 minutes for 500 iterations with batch size 6 [20 X 144 X 144] patch size. You can achieve reasonable results within 4 hours (more gpu will also reduce your training time.)

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

## Notes
Results for brats2018 will be updated and more experiments will be included. [2018/7/31]