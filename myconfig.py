import argparse
import os

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--multiple_device_id", type=tuple, default=(0,1))
parser.add_argument("--num_works", type=int, default=8)
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')

# Data
parser.add_argument("--dataset", type=str, default='cxr', help='can be bch, jsrt, montgomery')
parser.add_argument("--data_dir", type=str, default="/data/users/dewenzeng/data/cxr/supervised/")
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument("--enable_few_data", default=False, action='store_true')
parser.add_argument('--sampling_k', type=int, default=10)
parser.add_argument('--cross_vali_num', type=int, default=5)

# Model
parser.add_argument("--model_name", type=str, default='unet', help='can be unet or deeplab')
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--classes", type=int, default=4)
parser.add_argument("--initial_filter_size", type=int, default=32)

# Train
parser.add_argument("--experiment_name", type=str, default="supervised_cxr")
parser.add_argument("--restart", default=False, action='store_true')
parser.add_argument("--use_vanilla", default=False, action='store_true', help='whether use vanilla moco or simclr')
parser.add_argument("--pretrained_model_path", type=str, default='/afs/crc.nd.edu/user/d/dzeng2/UnsupervisedSegmentation/results/contrast_2020-09-30_02-37-23/model/latest.pth')
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--min_lr", type=float, default=1e-5)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--lr_scheduler", type=str, default='cos')
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--contrastive_method", type=str, default='cl_tci_simclr', help='simclr or cl_tci_simclr')
parser.add_argument("--pretext_method", type=str, default='rotation', help='rotation or pirl')

# Loss
parser.add_argument("--temp", type=float, default=0.1)

# Test
parser.add_argument("--step_size", type=float, default=0.5)

def save_args(obj, defaults, kwargs):
    for k,v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)

def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    return config
