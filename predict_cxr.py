import torch
import cv2
import os
import matplotlib.pyplot as plt
from network.unet2d import UNet2D
from dataset.utils import *
from PIL import Image
from utils import *

if not os.path.exists('./results_prediction/'):
  os.mkdir('./results_prediction/')
save_dir = './results_prediction/'+'mont_s20_random'
if not os.path.exists(save_dir):
  os.mkdir(save_dir)
if not os.path.exists(os.path.join(save_dir, 'mask')):
  os.mkdir(os.path.join(save_dir, 'mask'))
if not os.path.exists(os.path.join(save_dir, 'contour')):
  os.mkdir(os.path.join(save_dir, 'contour'))

patch_size = 256
def resize(image):
  image = pad_if_not_square(image)
  return cv2.resize(image, (patch_size, patch_size), cv2.INTER_CUBIC)
      
# define model
model = UNet2D(in_channels=1, initial_filter_size=32, kernel_size=3, classes=3)
# load model we use one of the cross validation model and test its validation fold
model_path = '/data/users/dewenzeng/code/cxr_segmentation_rmyy/results/supervised_mont_unet_s20_2021-07-02_05-26-37/cross_val_0/model/latest.pth'
dict = torch.load(model_path, map_location=lambda storage, loc: storage)
save_model = dict["net"]
model.load_state_dict(save_model)

dataset = 'mont'
# initialize dataloader
if dataset == 'cxr':
    _, val_keys = get_split(0, 5)
    data_dir = '/data/users/dewenzeng/data/cxr/supervised/'
    image_files = []
    label_files = []
    for key in val_keys:
        image_files.append(os.path.join(data_dir,'image','image_%03d'%key+'.jpg'))
        label_files.append(os.path.join(data_dir,'label','label_%03d'%key+'.png'))
elif dataset == 'jsrt':
    _, val_keys = get_split_jsrt(0, 5)
    data_dir = '/data/users/dewenzeng/data/jsrt/'
    image_files = []
    label_files = []
    for key in val_keys:
        image_files.append(os.path.join(data_dir,'image','image_%03d'%key+'.jpg'))
        label_files.append(os.path.join(data_dir,'label_lung_heart','label_%03d'%key+'.png'))
elif dataset == 'mont':
    _, val_keys = get_split_mont(0, 5)
    data_dir = '/data/users/dewenzeng/data/montgomery/'
    image_files = []
    label_files = []
    for key in val_keys:
        image_files.append(os.path.join(data_dir,'image','image_%03d'%key+'.png'))
        label_files.append(os.path.join(data_dir,'label','label_%03d'%key+'.png'))

for i in range(len(image_files)):
    print(f'processing image_{val_keys[i]}')
    original_image = cv2.imread(image_files[i], 0)
    original_label = Image.open(label_files[i])
    original_label = np.asarray(original_label).astype(np.uint8)
    original_image = original_image / 255.0
    image = resize(original_image)
    # print(f'image:{image.shape}')
    image_var = torch.from_numpy(image)
    # make prediction
    image_var = image_var.unsqueeze(dim=0).unsqueeze(dim=0).float()
    out = model(image_var)
    out = torch.argmax(torch.nn.functional.softmax(out, dim=1),dim=1).squeeze().cpu().numpy()

    target_width, target_height = original_image.shape
    max_one = max([target_width,target_height])
    out_resized = cv2.resize(out.astype(np.uint8), (max_one, max_one), interpolation=cv2.INTER_NEAREST)
    out = out_resized[int(max_one/2.0)-int(target_width/2.0):int(max_one/2.0)-int(target_width/2.0)+target_width,int(max_one/2.0)-int(target_height/2.0):int(max_one/2.0)-int(target_height/2.0)+target_height]

    original_image = (original_image * 255).astype(np.uint8)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # mask output lung on the original image
    original_image_copy = original_image.copy()
    left_lung = (out == 1).astype(np.uint8) * 255
    right_lung = (out == 2).astype(np.uint8) * 255

    _, threshed_left_lung = cv2.threshold(left_lung.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
    contours_left_lung, _ = cv2.findContours(threshed_left_lung, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    _, threshed_right_lung = cv2.threshold(right_lung.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
    contours_right_lung, _ = cv2.findContours(threshed_right_lung, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours_left_lung:
        cv2.drawContours(original_image_copy, contour, -1, (0, 255, 0), thickness=30)
    for contour in contours_right_lung:
        cv2.drawContours(original_image_copy, contour, -1, (255, 0, 0), thickness=30)

    cv2.imwrite(os.path.join(save_dir, 'contour', 'image_%03d'%val_keys[i]+'.jpg'), original_image_copy)