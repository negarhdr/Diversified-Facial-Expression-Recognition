import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import datetime
from model.diversified_manet import manet
from utils import udata
from torch.utils.data import DataLoader
from PIL import Image


now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
checkpoint_path = ''

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../FER_data/FER_plus/Dataset/', help='path to dataset')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/FER/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/FER/'+time_str+'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=checkpoint_path, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--classes', type=int, default=8)
args = parser.parse_args()
print('beta', args.beta)


class BranchDiversity(nn.Module):
    def __init__(self, ):
        super(BranchDiversity, self).__init__()
        self.direct_div = 0
        self.det_div = 0
        self.logdet_div = 0
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, type='spatial'):

        num_branches = x.size(0)
        gamma = 10
        snm = torch.zeros((num_branches, num_branches))

        # Spatial attn diversity
        if type == 'spatial':  # num_patches x batch_size x 512 x 7 x 7
            # diversity between spatial attention heads
            x = torch.mean(x, 2)  # num_patches x batch_size x 7 x 7
            for i in range(num_branches):
                for j in range(num_branches):
                    if i != j:
                        diff = torch.exp(-1 * gamma * torch.sum(torch.square(x[i, :, :, :] - x[j, :, :, :]), (1, 2)))
                        # batch_size
                        diff = torch.mean(diff)  # (1/num_branches) * torch.sum(diff)  # 1
                        snm[i, j] = diff
            self.direct_div = torch.sum(snm)
            self.det_div = -1 * torch.det(snm)
            self.logdet_div = -1 * torch.logdet(snm)

        # Channel attn diversity
        elif type == 'channel':  # num_patch x batch_size x 512 x 7 x 7
            x = self.avgpool(x)  #
            x = x.view(num_branches, -1, 512)  # num_patch x batch_size x 512

            # diversity between channels of attention heads
            for i in range(num_branches):
                for j in range(num_branches):
                    if i != j:
                        diff = torch.exp(
                            -1 * gamma * torch.sum(torch.square(x[i, :, :] - x[j, :, :]), 1))  # batch_size
                        diff = torch.mean(diff)  # (1/num_branches) * torch.sum(diff)  # 1
                        snm[i, j] = diff
            self.direct_div = torch.sum(snm)
            self.det_div = -1 * torch.det(snm)
            self.logdet_div = -1 * torch.logdet(snm)

        return self


class FeatDiversity(nn.Module):
    def __init__(self, ):
        super(FeatDiversity, self).__init__()
        self.direct_div = 0
        self.det_div = 0
        self.logdet_div = 0

    def forward(self, x):

        num_branches = x.size(0)  # num_branches x batch_size x 512
        gamma = 10
        snm = torch.zeros((num_branches, num_branches))
        # diversity between channels of attention heads
        for i in range(num_branches):
            for j in range(num_branches):
                if i != j:
                    diff = torch.exp(
                        -1 * gamma * torch.sum(torch.square(x[i, :, :] - x[j, :, :]), 1))  # batch_size
                    diff = torch.mean(diff)  # (1/num_branches) * torch.sum(diff)  # 1
                    snm[i, j] = diff
        self.direct_div = torch.sum(snm)
        self.det_div = -1 * torch.det(snm)
        self.logdet_div = -1 * torch.logdet(snm)

        return self


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model = manet()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load('./checkpoint/MS-Celeb-1M/Pretrained_on_MSCeleb.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    model.load_state_dict(pre_trained_dict)
    model.module.fc_1 = torch.nn.Linear(512, 8).cuda()
    model.module.fc_2 = torch.nn.Linear(512, 8).cuda()

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda()
    # diversity between the features of two branches
    criterion_bdiv = BranchDiversity()
    # diversity between the spatial and channel features of different patches in the local branch
    criterion_fdiv = FeatDiversity()
    optimizer = torch.optim.SGD(model.parameters(),  args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    recorder = RecorderMeter(args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # loading FER+ dataset
    # Define transforms
    data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomAffine(degrees=30,
                                               translate=(.1, .1),
                                               scale=(1.0, 1.25),
                                               resample=Image.BILINEAR)]

    # load train set
    train_data = udata.FERplus(idx_set=0,
                               max_loaded_images_per_label=5000,
                               transforms=transforms.Compose(data_transforms),
                               base_path_to_FER_plus=args.data)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)

    # load val data
    val_data = udata.FERplus(idx_set=1,
                             max_loaded_images_per_label=100000,
                             transforms=None,
                             base_path_to_FER_plus=args.data)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=8)

    if args.evaluate:
        validate(val_loader, model, criterion_ce, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'FER_plus_div_log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion_ce, criterion_bdiv, criterion_fdiv, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion_ce, args)

        scheduler.step()

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best, args)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')


def train(train_loader, model, criterion_ce, criterion_bdiv, criterion_fdiv, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        # compute output
        output1, output2, global_feat_b12, local_feat_b1 = model(images)
        output = (args.beta * output1) + ((1-args.beta) * output2)
        loss = (args.beta * criterion_ce(output1, target)) + ((1-args.beta) * criterion_ce(output2, target))
        loss += criterion_bdiv(local_feat_b1, type='spatial').det_div
        loss += criterion_bdiv(local_feat_b1, type='channel').det_div
        loss += criterion_fdiv(global_feat_b12).det_div

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion_ce, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output1, output2, global_feat_b12, local_feat_b1 = model(images)
            output = (args.beta * output1) + ((1-args.beta) * output2)
            loss = (args.beta * criterion_ce(output1, target)) + ((1 - args.beta) * criterion_ce(output2, target))

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()
