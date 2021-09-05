import numpy as np
import os
import shutil
import argparse
from PIL import Image
from skimage.io import imsave

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='d:/data/MontgomerySet/CXR_png')
parser.add_argument("--label_dir", type=str, default='d:/data/MontgomerySet/ManualMask/')
parser.add_argument("--result_dir", type=str, default='d:/data/MontgomerySet/converted_Montgomery/')
args = parser.parse_args()

data_dir = args.data_dir
label_dir = args.label_dir
result_dir = args.result_dir

# check if exist
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
image_result_dir = os.path.join(result_dir, 'image')
label_result_dir = os.path.join(result_dir, 'label')
# check if exist
if not os.path.exists(image_result_dir):
    os.mkdir(image_result_dir)
if not os.path.exists(label_result_dir):
    os.mkdir(label_result_dir)

n = 0
files = os.listdir(data_dir)
files.sort()
for file in files:
    image_left_lung = Image.open(os.path.join(label_dir, 'leftMask', file))
    image_left_lung = np.asarray(image_left_lung).astype(np.uint8)
    image_right_lung = Image.open(os.path.join(label_dir, 'rightMask', file))
    image_right_lung = np.asarray(image_right_lung).astype(np.uint8)
    mask = np.zeros_like(image_left_lung)
    mask = mask.astype(np.uint8)
    mask[image_left_lung==1] = 1
    mask[image_right_lung==1] = 2
    imsave(os.path.join(label_result_dir,'label_%03d'%n+'.png'), mask)
    shutil.copy(os.path.join(data_dir, file), os.path.join(image_result_dir, 'image_%03d'%n+'.png'))
    n = n + 1

print('finished...')