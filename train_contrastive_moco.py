import os
from datetime import datetime
from utils import *
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from network.unet2d import UNet2D_contrastive
from network.deeplabv3plus import deeplabv3_resnet50_contrast
from dataset.bch import BCH
from myconfig import get_config
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from PytorchExperimentLogger import PytorchExperimentLogger
from network.moco import MoCo

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
    logger.print(f"the model will run on device {args.multiple_device_id}")

    # create model
    logger.print("creating model ...")
    if args.model_name == 'unet':
        model = MoCo(UNet2D_contrastive, dim=args.classes, K=3072, m=0.999, T=args.temp)
    elif args.model_name == 'deeplab':
        model = MoCo(deeplabv3_resnet50_contrast, dim=args.classes, K=3072, m=0.999, T=args.temp, weight_func=args.weight_func)
    model.to(args.device)
    model = torch.nn.DataParallel(model, device_ids=args.multiple_device_id)

    num_parameters = sum([l.nelement() for l in model.module.parameters()])
    logger.print(f"number of parameters: {num_parameters}")

    train_dataset = BCH(keys=None, purpose='train', args=args)
    logger.print('training data dir ' + train_dataset.data_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, drop_last=True)

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), warmup_epochs=10)

    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, epoch, optimizer, scheduler, logger, args)

        logger.print('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     .format(epoch + 1, train_loss=train_loss))

        writer.add_scalar('training_loss', train_loss, epoch)

        # save model
        save_dict = {"net": model.module.state_dict()}
        torch.save(save_dict, os.path.join(args.model_result_dir, "latest.pth"))

def train(data_loader, model, epoch, optimizer, scheduler, logger, args):
    model.train()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        scheduler(optimizer, batch_idx, epoch)
        img1, img2, label = tup
        # We are training a image reconstruction network, the targets are the original inputs.
        image1_var = Variable(img1.float(), requires_grad=False).to(args.device)
        image2_var = Variable(img2.float(), requires_grad=False).to(args.device)
        label = Variable(label.long()).to(args.device)
        loss = model(im_q=image1_var, im_k=image2_var, pseudo_label=label, vanilla=args.use_vanilla)
        loss = loss.mean()
        bsz = img1.shape[0]
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.print(f"epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}")
    return losses.avg

if __name__ == '__main__':
    main()