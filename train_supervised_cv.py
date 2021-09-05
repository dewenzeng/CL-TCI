import os
from datetime import datetime
from utils import *
import torch.backends.cudnn as cudnn
import random
from torch.autograd import Variable
from network.unet2d import UNet2D
from network.deeplabv3plus import deeplabv3_resnet50
from dataset.bch import BCH
from dataset.jsrt import JSRT
from dataset.montgomery import Montgomery
import torch.nn.functional as F
from metrics import SegmentationMetric
from myconfig import get_config
from batchgenerators.utilities.file_and_folder_operations import *
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from PytorchExperimentLogger import PytorchExperimentLogger

def run(fold, writer, args):

    maybe_mkdir_p(os.path.join(args.save_path, 'cross_val_'+str(fold)))
    logger = PytorchExperimentLogger(os.path.join(args.save_path, 'cross_val_'+str(fold)), "elog", ShowTerminal=True)
    model_result_dir = join(args.save_path, 'cross_val_'+str(fold), 'model')
    maybe_mkdir_p(model_result_dir)
    args.model_result_dir = model_result_dir

    # setup cuda
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.print(f"the model will run on device: {args.device}")
    torch.manual_seed(args.seed)
    if 'cuda' in str(args.device):
        torch.cuda.manual_seed_all(args.seed)
    
    # create model
    logger.print("creating model ...")
    if args.model_name=='unet':
        model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes)
        if args.restart:
            logger.print('loading from saved model ' + args.pretrained_model_path)
            dict = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            save_model = dict["net"]
            model_dict = model.state_dict()
            # we only need to load the parameters of the encoder
            state_dict = {k.replace('module.',''): v for k, v in save_model.items() if "encoder" in k}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
    elif args.model_name=='moco_unet':
        model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes)
        if args.restart:
            logger.print('loading from saved model ' + args.pretrained_model_path)
            dict = torch.load(args.pretrained_model_path,
                            map_location=lambda storage, loc: storage)
            save_model = dict["net"]
            for k in list(save_model.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder_q'):
                    # remove prefix
                    save_model[k[len("encoder_q."):]] = save_model[k]
                # delete renamed or unused k
                del save_model[k]
            model_dict = model.state_dict()
            # we only need to load the parameters of the encoder
            state_dict = {k: v for k, v in save_model.items() if "encoder" in k}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
    elif args.model_name=='deeplab':
        model = deeplabv3_resnet50(num_classes=args.classes, pretrained_backbone=False)
        if args.restart:
            logger.print('loading from saved model ' + args.pretrained_model_path)
            dict = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            save_model = dict["net"]
            model_dict = model.state_dict()
            # we only need to load the parameters of the encoder
            state_dict = {k.replace('module.',''): v for k, v in save_model.items() if "backbone" in k}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
    elif args.model_name=='moco_deeplab':
        model = deeplabv3_resnet50(num_classes=args.classes, pretrained_backbone=False)
        if args.restart:
            logger.print('loading from saved model ' + args.pretrained_model_path)
            dict = torch.load(args.pretrained_model_path,
                            map_location=lambda storage, loc: storage)
            save_model = dict["net"]
            for k in list(save_model.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder_q'):
                    # remove prefix
                    save_model[k[len("encoder_q."):]] = save_model[k]
                # delete renamed or unused k
                del save_model[k]
            model_dict = model.state_dict()
            # we only need to load the parameters of the encoder
            state_dict = {k.replace('module.',''): v for k, v in save_model.items() if "backbone" in k}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

    model.to(args.device)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.print(f"number of parameters: {num_parameters}")

    # initialize dataloader
    if args.dataset == 'bch':
        train_keys, val_keys = get_split_bch(fold, args.cross_vali_num)
        # now random sample train_keys
        if args.enable_few_data:
            random.seed(args.seed)
            train_keys = random.sample(list(train_keys), k=args.sampling_k)
        train_dataset = BCH(train_keys, purpose='train', args=args)
        logger.print('training data dir '+train_dataset.data_dir)
        validate_dataset = BCH(val_keys, purpose='val', args=args)
    elif args.dataset == 'jsrt':
        train_keys, val_keys = get_split_jsrt(fold, args.cross_vali_num)
        if args.enable_few_data:
            random.seed(args.seed)
            train_keys = random.sample(list(train_keys), k=args.sampling_k)
        train_dataset = JSRT(train_keys, purpose='train', args=args)
        logger.print('training data dir '+train_dataset.data_dir)
        validate_dataset = JSRT(val_keys, purpose='val', args=args)
    elif args.dataset == 'montgomery':
        train_keys, val_keys = get_split_mont(fold, args.cross_vali_num)
        if args.enable_few_data:
            random.seed(args.seed)
            train_keys = random.sample(list(train_keys), k=args.sampling_k)
        train_dataset = Montgomery(train_keys, purpose='train', args=args)
        logger.print('training data dir '+train_dataset.data_dir)
        validate_dataset = Montgomery(val_keys, purpose='val', args=args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=10, shuffle=False, num_workers=8, drop_last=False)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), min_lr=args.min_lr)

    best_dice = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, epoch, optimizer, scheduler, logger, args)
        writer.add_scalar('training_loss_fold_'+str(fold), train_loss, epoch)
        if (epoch % 2 == 0):
            # evaluate for one epoch
            val_dice = validate(validate_loader, model, epoch, logger, args)

            logger.print('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Validation Dice {val_dice:.4f} \t'
                         .format(epoch + 1, train_loss=train_loss, val_dice=val_dice))

            if best_dice < val_dice:
                best_dice = val_dice
                # save best model
                save_dict = {"net": model.state_dict()}
                torch.save(save_dict, os.path.join(args.model_result_dir, "best.pth"))
            writer.add_scalar('validate_dice_fold_'+str(fold), val_dice, epoch)
            writer.add_scalar('best_dice_fold'+str(fold), best_dice, epoch)
            writer.add_scalar('learning_rate_fold_'+str(fold), optimizer.param_groups[0]['lr'], epoch)

            # save model
            save_dict = {"net": model.state_dict()}
            torch.save(save_dict, os.path.join(args.model_result_dir, "latest.pth"))

def train(data_loader, model, criterion, epoch, optimizer, scheduler, logger, args):
    model.train()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        img, label = tup
        image_var = Variable(img.float(), requires_grad=False).to(args.device)
        label = Variable(label.long()).to(args.device)
        scheduler(optimizer, batch_idx, epoch)
        x_out = model(image_var)
        loss = criterion(x_out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), image_var.size(0))
        logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}")
    return losses.avg

def validate(data_loader, model, epoch, logger, args):
    model.eval()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()
    with torch.no_grad():
        for batch_idx, tup in enumerate(data_loader):
            img, label = tup
            # We are training a image reconstruction network, the targets are the original inputs.
            img_var = Variable(img.float(), requires_grad=False).to(args.device)
            target_var = Variable(label.long()).to(args.device)
            x_out = model(img_var)
            x_out = F.softmax(x_out, dim=1)
            metric_val.update(target_var.long().squeeze(), x_out)
            pixAcc, mIoU, Dice = metric_val.get()
            logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, mean Dice:{Dice}")
    pixAcc, mIoU, Dice = metric_val.get()
    return Dice

if __name__ == '__main__':
    args = get_config()
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    writer = SummaryWriter('runs/' + args.experiment_name + args.save)
    for i in range(0, args.cross_vali_num):
        run(i, writer, args)