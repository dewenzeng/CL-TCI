from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.data.dataset import Dataset
from PIL import Image
from .utils import *
from .augmentation import *

class BCH(Dataset):

    def __init__(self, keys, purpose, args):

        self.data_dir = args.data_dir
        self.purpose = purpose
        self.do_contrast = args.do_contrast
        if self.do_contrast:
            self.image_files = []
            # the pseudo_labels is the 
            self.pseudo_labels = []
            # save all image paths in a file list
            patients = os.listdir(self.data_dir)
            patients.sort()
            for i in range(len(patients)):
                images = os.listdir(os.path.join(self.data_dir, patients[i]))
                images.sort()
                for image in images:
                    self.image_files.append(os.path.join(self.data_dir, patients[i],image))
                    self.pseudo_labels.append(i)

        else:
            self.image_files = []
            self.label_files = []
            for key in keys:
                self.image_files.append(os.path.join(self.data_dir,'image','image_%03d'%key+'.jpg'))
                self.label_files.append(os.path.join(self.data_dir,'label','label_%03d'%key+'.png'))
        self.patch_size = args.patch_size

    def __getitem__(self, index):

        if self.do_contrast:
            return self.prepare_for_contrast(index)
        else:
            return  self.prepare_for_supervised(index)
    
    def prepare_for_contrast(self, index):

        img = Image.open(self.image_files[index]).convert('L')
        pseudo_label = self.pseudo_labels[index]

        img = np.asarray(img).astype(np.uint8)
        img = pad_if_not_square(img)
        dummy_label = np.zeros_like(img)

        train_transform = Compose([
            AdjustSaturation(0.4),
            AdjustContrast(0.4),
            AdjustBrightness(0.4),
            AdjustHue(0.4),
            AdjustGamma(0.4),
            RandomTranslate(offset=(0.1, 0.1)),
            RandomRotate(degree=10),
            RandomSizedCrop(size=self.patch_size,scale=(0.95, 1.)),
            ToTensor(),
        ])

        img1, _ = train_transform(img,dummy_label)
        img2, _ = train_transform(img,dummy_label)

        return img1, img2, pseudo_label

    def prepare_for_supervised(self, index):

        img = Image.open(self.image_files[index]).convert('L')
        label = Image.open(self.label_files[index])

        img = np.asarray(img).astype(np.uint8)
        label = np.asarray(label).astype(np.uint8)

        # 3 is the heart label, we just do lung segmentation here.
        label[label==1] = 1
        label[label==2] = 2
        label[label==3] = 0

        img = pad_if_not_square(img)
        label = pad_if_not_square(label)

        # if we have data augmentation
        train_transform = Compose([
            AdjustSaturation(0.4),
            AdjustContrast(0.4),
            AdjustBrightness(0.4),
            AdjustHue(0.4),
            RandomTranslate(offset=(0.1, 0.1)),
            RandomRotate(degree=30),
            RandomSizedCrop(size=self.patch_size,scale=(0.95, 1.)),
            ToTensor(),
        ])

        test_transform = Compose([
            Scale(size=self.patch_size),
            ToTensor()
        ])

        if self.purpose == 'train':
            img, label = train_transform(img, label)
        else:
            img, label = test_transform(img, label)
        
        return img, label

    def __len__(self):
        return len(self.image_files)
