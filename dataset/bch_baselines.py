import cv2
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.data.dataset import Dataset
from PIL import Image
from .utils import *
from .augmentation import *
from torchvision import transforms

class StackTransform(object):
    """transform a group of images independently"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        return torch.stack([self.transform(crop) for crop in imgs])


class JigsawCrop(object):
    """Jigsaw style crop"""
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

    def __call__(self, img):
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size])
        # crops = [Image.fromarray(crop) for crop in crops]
        return crops

class BCH_baseline(Dataset):

    def __init__(self, args):

        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.pretext_method = args.pretext_method
        self.image_files = []
        # save all image paths in a file list
        patients = os.listdir(self.data_dir)
        patients.sort()
        for i in range(len(patients)):
            images = os.listdir(os.path.join(self.data_dir, patients[i]))
            images.sort()
            for image in images:
                self.image_files.append(os.path.join(self.data_dir, patients[i],image))
        print(f'dataset length: {len(self.image_files)}')

    def __getitem__(self, index):
        if self.pretext_method == 'rotation':
            img = Image.open(self.image_files[index]).convert('L')
            img, img_90, img_180, img_270 = self.prepare_rotation(img)
            return img, img_90, img_180, img_270, 0, 1, 2, 3
        else:
            img = Image.open(self.image_files[index]).convert('L')
            img, img_jig = self.prepare_jigsaw(img)
            return img, img_jig, index

    def __len__(self):
        return len(self.image_files)

    # use this function for jigsaw pretext task
    def prepare_jigsaw(self, img):
        # resize image
        img = np.asarray(img).astype(np.uint8)
        img = pad_if_not_square(img)
        img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation = cv2.INTER_AREA)
        # img = img / 255.0
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        jig_transform = transforms.Compose([
            JigsawCrop(n_grid=3, img_size=self.patch_size, crop_size=80),
            StackTransform(transforms.Compose([
                    transforms.ToTensor(),
            ]))
        ])
        # print(f'img:{img.min()}')
        img_jig = jig_transform(255 * img)
        img = train_transform(img)
        return img, img_jig

    # use this function for rotation pretext task
    def prepare_rotation(self, img):
        # resize image
        img = np.asarray(img).astype(np.uint8)
        img = pad_if_not_square(img)
        img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation = cv2.INTER_AREA)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        img_90 = torch.from_numpy(np.rot90(img,k=1,axes=(1,2)).copy())
        img_180 = torch.from_numpy(np.rot90(img,k=2,axes=(1,2)).copy())
        img_270 = torch.from_numpy(np.rot90(img,k=3,axes=(1,2)).copy())
        return img, img_90, img_180, img_270
