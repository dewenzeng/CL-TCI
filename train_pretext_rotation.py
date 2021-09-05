import os
from datetime import datetime
from utils import *
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from network.unet2d import UNet2D_classification
from network.deeplabv3plus import deeplabv3_resnet50_classification
from dataset.bch_baselines import BCH_baseline
from myconfig import get_config
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from PytorchExperimentLogger import PytorchExperimentLogger

def main():
    # initialize config
    args = get_config()

    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = PytorchExperimentLogger(save_path, "elog", ShowTerminal=True)
    model_result_dir = os.path.join(save_path, 'model')
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)
    args.model_result_dir = model_result_dir

    logger.print(f"saving to {save_path}")
    writer = SummaryWriter('runs/' + args.experiment_name + args.save)

    # setup cuda
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.print(f"the model will run on device {args.device}")

    # create model
    logger.print("creating model ...")
    if args.model_name == 'unet':
        model = UNet2D_classification(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes, do_instancenorm=True)
    elif args.model_name == 'deeplab':
        model = deeplabv3_resnet50_classification(classes=args.classes, pretrained_backbone=False)
    model.to(args.device)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.print(f"number of parameters: {num_parameters}")

    # does not matter with the keys and purpose when we do contrastive learning.
    train_dataset = BCH_baseline(args=args)
    logger.print('training data dir ' + train_dataset.data_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, drop_last=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))

    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, criterion, epoch, optimizer, scheduler, logger, args)

        logger.print('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     .format(epoch + 1, train_loss=train_loss))

        writer.add_scalar('training_loss', train_loss, epoch)

        # save model
        save_dict = {"net": model.state_dict()}
        torch.save(save_dict, os.path.join(args.model_result_dir, "latest.pth"))

def train(data_loader, model, criterion, epoch, optimizer, scheduler, logger, args):
    model.train()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        scheduler(optimizer, batch_idx, epoch)
        img, img_90, img_180, img_270, label, label_90, label_180, label_270 = tup
        img_all = torch.cat((img, img_90, img_180, img_270),dim=0)
        label_all = torch.cat((label, label_90, label_180, label_270),dim=0)
        image_var = Variable(img_all.float(), requires_grad=False).to(args.device)
        label_var = Variable(label_all.long()).to(args.device)
        out = model(image_var)
        bsz = image_var.shape[0]
        loss = criterion(out, label_var)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}")
    return losses.avg

if __name__ == '__main__':
    main()