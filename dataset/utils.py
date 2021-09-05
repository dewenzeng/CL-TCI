import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

# Convolutional Neural Network With Shape Prior Applied to Cardiac MRI Segmentation
def to_categorical(matrix, nb_classes):
    """ Transform a matrix containing interger label class into a matrix containing categorical class labels.
    The last dim of the matrix should be the category (classes).

    Args:
        matrix: ndarray, A numpy matrix to convert into a categorical matrix.
        nb_classes: int, number of classes

    Returns:
        A numpy array representing the categorical matrix of the input.
    """
    return (matrix == np.arange(nb_classes)[np.newaxis, np.newaxis, :]).astype(np.uint32)

# image = np.random.randint(0,4,(4,4),np.uint8)
# image = image[np.newaxis,...,np.newaxis]
# print(f'image before:{image[0,:,:,0]}')
# image = to_categorical(image,4)
# print(f'image after:{image[0,:,:,0]}')
# print(f'image after:{image[0,:,:,1]}')

def pad_if_too_small(data, sz):
  reshape = (len(data.shape) == 2)
  if reshape:
    h, w = data.shape
    data = data.reshape((h, w, 1))

  h, w, c = data.shape

  if not (h >= sz and w >= sz):
    # img is smaller than sz
    # we are missing by at least 1 pixel in at least 1 edge
    new_h, new_w = max(h, sz), max(w, sz)
    new_data = np.zeros([new_h, new_w, c], dtype=data.dtype)

    # will get correct centre, 5 -> 2
    centre_h, centre_w = int(new_h / 2.), int(new_w / 2.)
    h_start, w_start = centre_h - int(h / 2.), centre_w - int(w / 2.)

    new_data[h_start:(h_start + h), w_start:(w_start + w), :] = data
  else:
    new_data = data
    new_h, new_w = h, w

  if reshape:
    new_data = new_data.reshape((new_h, new_w))

  return new_data

def pad_if_not_square(orig_data):
  w, h = orig_data.shape
  if w == h:
    return orig_data
  elif w > h:
    return pad_if_too_small(orig_data,w)
  else:
    return pad_if_too_small(orig_data,h)

def pad_and_or_crop(orig_data, sz, mode=None, coords=None):
  data = pad_if_too_small(orig_data, sz)

  reshape = (len(data.shape) == 2)
  if reshape:
    h, w = data.shape
    data = data.reshape((h, w, 1))

  h, w, c = data.shape
  if mode == "centre":
    h_c = int(h / 2.)
    w_c = int(w / 2.)
  elif mode == "fixed":
    assert (coords is not None)
    h_c, w_c = coords
  elif mode == "random":
    h_c_min = int(sz / 2.)
    w_c_min = int(sz / 2.)

    if sz % 2 == 1:
      h_c_max = h - 1 - int(sz / 2.)
      w_c_max = w - 1 - int(sz / 2.)
    else:
      h_c_max = h - int(sz / 2.)
      w_c_max = w - int(sz / 2.)

    h_c = np.random.randint(low=h_c_min, high=(h_c_max + 1))
    w_c = np.random.randint(low=w_c_min, high=(w_c_max + 1))

  h_start = h_c - int(sz / 2.)
  w_start = w_c - int(sz / 2.)
  data = data[h_start:(h_start + sz), w_start:(w_start + sz), :]

  if reshape:
    data = data.reshape((sz, sz))

  return data, (h_c, w_c)

def convert_to_time_string(str):
  date_str = datetime(year=int(str[0:4]),month=int(str[4:6]),day=int(str[6:8]),hour=int(str[9:11]),minute=int(str[11:13]),second=int(str[13:15]))
  return date_str

def perform_affine_tf(data, tf_matrices):
  # expects 4D tensor, we preserve gradients if there are any

  n_i, k, h, w = data.shape
  n_i2, r, c = tf_matrices.shape
  assert (n_i == n_i2)
  assert (r == 2 and c == 3)

  grid = F.affine_grid(tf_matrices, data.shape, align_corners=False)  # output should be same size
  data_tf = F.grid_sample(data, grid,
                          padding_mode="zeros", align_corners=False)  # this can ONLY do bilinear

  return data_tf

def random_affine(img, min_rot=None, max_rot=None, min_shear=None,
                  max_shear=None, min_scale=None, max_scale=None):
    # Takes and returns torch cuda tensors with channels 1st (1 img)
    # rot and shear params are in degrees
    # tf matrices need to be float32, returned as tensors
    # we don't do translations

    # https://github.com/pytorch/pytorch/issues/12362
    # https://stackoverflow.com/questions/42489310/matrix-inversion-3-3-python
    # -hard-coded-vs-numpy-linalg-inv

    # https://github.com/pytorch/vision/blob/master/torchvision/transforms
    # /functional.py#L623
    # RSS(a, scale, shear) = [cos(a) *scale   - sin(a + shear) * scale     0]
    #                        [ sin(a)*scale    cos(a + shear)*scale     0]
    #                        [     0                  0          1]
    # used by opencv functional _get_affine_matrix and
    # skimage.transform.AffineTransform

    assert (len(img.shape) == 3)
    a = np.radians(np.random.rand() * (max_rot - min_rot) + min_rot)
    shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
    scale = np.random.rand() * (max_scale - min_scale) + min_scale

    affine1_to_2 = np.array([[np.cos(a) * scale, - np.sin(a + shear) * scale, 0.],
                            [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                            [0., 0., 1.]], dtype=np.float32)  # 3x3

    affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

    affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]  # 2x3
    affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2), \
                                torch.from_numpy(affine2_to_1)

    img = perform_affine_tf(img.unsqueeze(dim=0), affine1_to_2.unsqueeze(dim=0))
    img = img.squeeze(dim=0)

    return img, affine1_to_2, affine2_to_1

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        # use this function if image is grayscale
        plt.imshow(npimg[0,:,:],'gray')
        # use this function if image is RGB
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
