import glob
import os
import warnings
import shutil
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from nipype.interfaces.ants import N4BiasFieldCorrection

def N4BiasFieldCorrect(filename, output_filename):
    normalized = N4BiasFieldCorrection()
    normalized.inputs.input_image = filename
    normalized.inputs.output_image = output_filename
    normalized.run()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='training data path', default="/data/dataset/BRATS2018/training/")
    parser.add_argument('--out', help="output path", default="./N4_Normalized")
    parser.add_argument('--mode', help="output path", default="training")
    args = parser.parse_args()
    if args.mode == 'test':
        BRATS_data = glob.glob(args.data + "/*")
        patient_ids = [x.split("/")[-1] for x in BRATS_data]
        print("Processing Testing data ...")
        for idx, file_name in tqdm(enumerate(BRATS_data), total=len(BRATS_data)):
            mod = glob.glob(file_name+"/*.nii*")
            output_dir = "{}/test/{}/".format(args.out, patient_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for mod_file in mod:
                if 'flair' not in mod_file and 'seg' not in mod_file:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    N4BiasFieldCorrect(mod_file, output_path)
                else:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    shutil.copy(mod_file, output_path)
    else:
        HGG_data = glob.glob(args.data + "HGG/*")
        LGG_data = glob.glob(args.data + "LGG/*")
        hgg_patient_ids = [x.split("/")[-1] for x in HGG_data]
        lgg_patient_ids = [x.split("/")[-1] for x in LGG_data]
        print("Processing HGG ...")
        for idx, file_name in tqdm(enumerate(HGG_data), total=len(HGG_data)):
            mod = glob.glob(file_name+"/*.nii*")
            output_dir = "{}/HGG/{}/".format(args.out, hgg_patient_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for mod_file in mod:
                if 'flair' not in mod_file and 'seg' not in mod_file:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    N4BiasFieldCorrect(mod_file, output_path)
                else:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    shutil.copy(mod_file, output_path)
        print("Processing LGG ...")
        for idx, file_name in tqdm(enumerate(LGG_data), total=len(LGG_data)):
            mod = glob.glob(file_name+"/*.nii*")
            output_dir = "{}/LGG/{}/".format(args.out, lgg_patient_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for mod_file in mod:
                if 'flair' not in mod_file and 'seg' not in mod_file:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    N4BiasFieldCorrect(mod_file, output_path)
                else:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    shutil.copy(mod_file, output_path)



if __name__ == "__main__":
    main()