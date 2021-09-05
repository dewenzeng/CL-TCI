import torchvision
from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.data.dataset import Dataset
from PIL import Image
from .utils import *
from .augmentation import *

class Montgomery(Dataset):

    def __init__(self, keys, purpose, args):

        self.data_dir = args.data_dir
        self.purpose = purpose
        self.image_files = []
        self.label_files = []
        for key in keys:
            self.image_files.append(os.path.join(self.data_dir,'image','image_%03d'%key+'.png'))
            self.label_files.append(os.path.join(self.data_dir,'label','label_%03d'%key+'.png'))
        self.patch_size = args.patch_size

    def __getitem__(self, index):

        return  self.prepare_for_supervised(index)
    
    def prepare_for_supervised(self, index):

        img = Image.open(self.image_files[index]).convert('L')
        label = Image.open(self.label_files[index])

        img = np.asarray(img).astype(np.uint8)
        label = np.asarray(label).astype(np.uint8)

        img = pad_if_not_square(img)
        label = pad_if_not_square(label)

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

if __name__ == '__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="d:/data/MontgomerySet/converted/")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--classes", type=int, default=3)
    args = parser.parse_args()
    train_keys = np.arange(0,138)
    train_dataset = Montgomery(keys=train_keys,purpose='train',args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=5,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   drop_last=False)

    for batch_idx, tup in enumerate(train_dataloader):
        img, label = tup
        plt.figure(1)
        img_grid = torchvision.utils.make_grid(img)
        matplotlib_imshow(img_grid, one_channel=False)
        plt.figure(2)
        img_grid = torchvision.utils.make_grid(label.unsqueeze(dim=1))
        matplotlib_imshow(img_grid, one_channel=False)
        plt.show()
        break
