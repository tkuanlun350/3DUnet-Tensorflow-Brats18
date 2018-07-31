#####
# some functions are borrowed from https://github.com/taigw/brats17/
#####

import nibabel
import numpy as np
import random
import os
import SimpleITK as sitk
import pickle

def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def get_none_zero_region(im, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    # random normal too slow
    #out_random = np.random.normal(0, 1, size = volume.shape)
    out_random = np.zeros(volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def get_random_roi_sampling_center(input_shape, output_shape, sample_mode='full', bounding_box = None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)   
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center

def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def save_to_pkl(probs, filename, outdir=""):
    """
    probs => [155, 240, 240]
    """
    if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))

    with open("./{}/{}.pkl".format(outdir, filename), 'wb') as f:
        pickle.dump(probs, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_nii(im, filename, outdir="", mode="image", system="sitk"):
    """
    Save numpy array to nii.gz format to submit
    im: 3d numpy array ex: [155, 240, 240]
    """
    if system == "sitk":
        if mode == 'label':
            img = sitk.GetImageFromArray(im.astype(np.uint8))
        else:
            img = sitk.GetImageFromArray(im.astype(np.float32))
        if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))
        sitk.WriteImage(img, "./{}/{}.nii.gz".format(outdir, filename))
    else:
        img = np.rot90(im, k=2, axes= (1,2))
        OUTPUT_AFFINE = np.array(
                [[0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])
        if mode == 'label':
            img = nibabel.Nifti1Image(img.astype(np.uint8), OUTPUT_AFFINE)
        else:
            img = nibabel.Nifti1Image(img.astype(np.float32), OUTPUT_AFFINE)
        if not os.path.exists("./{}".format(outdir)):
            os.mkdir("./{}".format(outdir))
        nibabel.save(img, "./{}/{}.nii.gz".format(outdir, filename))
