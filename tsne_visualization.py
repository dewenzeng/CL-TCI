import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from network.deeplabv3plus import deeplabv3_resnet50_contrast
from network.unet2d import UNet2D_contrastive
from dataset.bch import BCH

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # device
    parser.add_argument("--device", type=str, default='cuda:0')
    # dataset
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--data_dir", type=str, default="d:/data/cxr/images3")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--classes", type=int, default=512)
    opt = parser.parse_args()
    return opt

opt = parse_option()
# opt.pretrained_model_path = './results/contrast_cxr_ours_unet_func1_2021-05-30_09-32-57/model/latest.pth'
opt.pretrained_model_path = './results/contrast_cxr_moco_v2_2021-06-16_10-37-14/model/latest.pth'

cl_type = 'moco'
# model = deeplabv3_resnet50_contrast(classes=512, pretrained_backbone=False)
model = UNet2D_contrastive(in_channels=1, initial_filter_size=32, kernel_size=3, classes=512, do_instancenorm=True)
# load the pretrained model
dict = torch.load(opt.pretrained_model_path, map_location=lambda storage, loc: storage)
state_dict = dict['net']
if cl_type == 'moco':
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q'):
            # remove prefix
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
model.load_state_dict(state_dict, strict=False)
model.to(opt.device)

dataset = BCH(keys=None, purpose='val', args=opt)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

features_list = []
labels_list = []

with torch.no_grad():
    for idx, (images, _, labels, _) in enumerate(data_loader):
        print(f'processing iteration: {idx}/{len(data_loader)}')
        images = images.float().to(opt.device)
        # forward
        f = model(images)
        features_list.append(f[0].data.cpu().numpy())
        labels_list.append(labels[0].data.cpu().numpy())
        # break

features_list = np.array(features_list)
labels_list = np.array(labels_list)
np.save('./tsne/features', features_list)
np.save('./tsne/labels', labels_list)
print(f'features_list:{features_list.shape}')

X = np.load('./tsne/features.npy')
y = np.load('./tsne/labels.npy')

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(6, 6))
color_map = plt.cm.get_cmap('viridis', 105)
for i in range(X_norm.shape[0]):
    # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
    #          fontdict={'weight': 'bold', 'size': 9})
    circle = plt.Circle([X_norm[i, 0], X_norm[i, 1]], radius=0.01, color=color_map(y[i]), fill=True)
    plt.gca().add_patch(circle)
    if np.random.random() < 0.05:
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), fontsize=10)
plt.xlim(0, 1)
plt.ylim(0, 1)
# c = [mpatches.Circle((0.5, 0.5), 0.01, facecolor=color_map(i)) for i in range(y.max()+1) ]
# test = [str(i) for i in range(y.max()+1) ]
# plt.gca().legend(c, test , bbox_to_anchor=(0, 1), loc='lower left', fontsize='small', ncol=7, handler_map={mpatches.Circle: HandlerEllipse()})
# plt.show()
plt.savefig('./tsne/tsne.png', dpi=300)