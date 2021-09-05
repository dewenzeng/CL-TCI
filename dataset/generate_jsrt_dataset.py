import cv2
import os
import shutil
import argparse
import numpy as np
from PIL import Image
from skimage.io import imsave

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='d:/data/JSRT/All247images/')
parser.add_argument("--label_dir", type=str, default='d:/data/JSRT/scratch/')
parser.add_argument("--result_dir", type=str, default='d:/data/JSRT/converted_JSRT/')
args = parser.parse_args()

data_dir = args.data_dir
label_dir = args.label_dir
result_dir = args.result_dir

# check if exist
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
image_result_dir = os.path.join(result_dir, 'original_image')
label_result_dir = os.path.join(result_dir, 'original_label')
# check if exist
if not os.path.exists(image_result_dir):
    os.mkdir(image_result_dir)
if not os.path.exists(label_result_dir):
    os.mkdir(label_result_dir)

files = os.listdir(data_dir)
for file in files:
    fid = open(os.path.join(data_dir, file), 'rb')
    dtype = np.dtype('>u2')
    shape = (2048, 2048)
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)
    threshold1 = 0
    threshold2 = 3000
    image[image<threshold1] = threshold1
    image[image>threshold2] = threshold2
    image = (image.astype(np.float) - threshold1) / threshold2
    image = 1 - image
    image = cv2.resize(image, (1024,1024), cv2.INTER_AREA)
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(image_result_dir, file.replace('.IMG','.jpg')), image)

# combine the annotations and save them as png, nodule and nonnodule annotation are in different folds
# nodule annotation
data_dir_left_lung = os.path.join(label_dir, 'fold1/masks/left_lung')
data_dir_right_lung = os.path.join(label_dir, 'fold1/masks/right_lung')
data_dir_heart = os.path.join(label_dir, 'fold1/masks/heart')
files = os.listdir(data_dir_left_lung)
for file in files:
    image_left_lung = Image.open(os.path.join(data_dir_left_lung, file))
    image_left_lung = np.asarray(image_left_lung) / 255
    image_right_lung = Image.open(os.path.join(data_dir_right_lung, file))
    image_right_lung = np.asarray(image_right_lung) / 255
    image_heart = Image.open(os.path.join(data_dir_heart, file))
    image_heart = np.asarray(image_heart) / 255
    mask = np.zeros_like(image_left_lung)
    mask = mask.astype(np.uint8)
    mask[image_right_lung==1] = 1
    mask[image_left_lung==1] = 2
    mask[image_heart==1] = 3
    imsave(os.path.join(label_result_dir, file.replace('.gif','.png')), mask)
# nonnodule annotation
data_dir_left_lung = os.path.join(label_dir, 'fold2/masks/left_lung')
data_dir_right_lung = os.path.join(label_dir, 'fold2/masks/right_lung')
data_dir_heart = os.path.join(label_dir, 'fold2/masks/heart')
files = os.listdir(data_dir_left_lung)
for file in files:
    image_left_lung = Image.open(os.path.join(data_dir_left_lung, file))
    image_left_lung = np.asarray(image_left_lung) / 255
    image_right_lung = Image.open(os.path.join(data_dir_right_lung, file))
    image_right_lung = np.asarray(image_right_lung) / 255
    image_heart = Image.open(os.path.join(data_dir_heart, file))
    image_heart = np.asarray(image_heart) / 255
    mask = np.zeros_like(image_left_lung)
    mask = mask.astype(np.uint8)
    mask[image_right_lung==1] = 1
    mask[image_left_lung==1] = 2
    mask[image_heart==1] = 3
    imsave(os.path.join(label_result_dir, file.replace('.gif','.png')), mask)

# rename all images
rename_image_result_dir = os.path.join(result_dir, 'image')
rename_label_result_dir = os.path.join(result_dir, 'label')
# check if exist
if not os.path.exists(rename_image_result_dir):
    os.mkdir(rename_image_result_dir)
if not os.path.exists(rename_label_result_dir):
    os.mkdir(rename_label_result_dir)
files = os.listdir(image_result_dir)
n = 0
for file in files:
    shutil.copy(os.path.join(image_result_dir, file), os.path.join(rename_image_result_dir, 'image_%03d'%n+'.jpg'))
    shutil.copy(os.path.join(label_result_dir, file.replace('jpg','png')), os.path.join(rename_label_result_dir, 'label_%03d'%n+'.png'))
    n = n + 1