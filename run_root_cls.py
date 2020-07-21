#-------------------------------------------------------------------------------
# Name:        run_root_cls.py
# Purpose:     Train a network (rootnet) to predict which joint is the root
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import os
import sys
sys.path.append("./")
import shutil
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch_geometric.data import DataLoader

from models.ROOT_GCN import ROOTNET
from utils.log_utils import AverageMeter
from utils.os_utils import isdir, mkdir_p, isfile
from torch.utils.tensorboard import SummaryWriter
from datasets.skeleton_dataset import GraphDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*gamma
    return lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def main(args):
    global device
    best_acc = 0.0

    # create checkpoint dir and log dir
    if not isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
    mkdir_p(args.checkpoint)
    if not args.resume:
        if isdir(args.logdir):
            shutil.rmtree(args.logdir)
        mkdir_p(args.logdir)

    # create model
    model = ROOTNET()
  
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    lr = args.lr
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_loader = DataLoader(GraphDataset(root=args.train_folder), batch_size=args.train_batch, shuffle=True)
    val_loader = DataLoader(GraphDataset(root=args.val_folder), batch_size=args.test_batch, shuffle=False)
    test_loader = DataLoader(GraphDataset(root=args.test_folder), batch_size=args.test_batch, shuffle=False)
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(test_loader, model, args)
        print('test_loss {:.8f}. test_acc: {:.6f}'.format(test_loss, test_acc))
        return

    logger = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)

        train_loss = train(train_loader, model, optimizer, args)
        #print('validating')
        val_loss, val_acc = test(val_loader, model, args)
        #print('testing')
        test_loss, test_acc = test(test_loader, model, args)

        print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch, train_loss))
        print('Epoch{:d}. val_loss: {:.6f}. val_acc: {:.6f}'.format(epoch, val_loss, val_acc))
        print('Epoch{:d}. test_loss: {:.6f}. test_acc: {:.6f}'.format(epoch, test_loss, test_acc))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc': best_acc,
                         'optimizer': optimizer.state_dict()}, is_best, checkpoint=args.checkpoint)

        info = {'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_acc,
                'test_loss': test_loss, 'test_accuracy': test_acc}
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch + 1)
    print("=> loading checkpoint '{}'".format(os.path.join(args.checkpoint, 'model_best.pth.tar')))
    checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
    best_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(os.path.join(args.checkpoint, 'model_best.pth.tar'), best_epoch))
    test_loss, test_acc = test(test_loader, model, args)
    print('Best epoch:\n test_loss {:8f} test_acc {:8f}'.format(test_loss, test_acc))


def train(train_loader, model, optimizer, args):
    global device
    model.train()  # switch to train mode
    loss_meter = AverageMeter()
    for data in train_loader:
        #print(data.name)
        data = data.to(device)
        optimizer.zero_grad()
        pre_label, label = model(data)
        loss_1 = torch.nn.functional.binary_cross_entropy_with_logits(pre_label, label, reduction='none')
        topk_val, _ = torch.topk(loss_1.view(-1), k=int(args.topk * len(pre_label)), dim=0, sorted=False)
        loss2 = topk_val.mean()
        #loss_3 = torch.nn.functional.binary_cross_entropy_with_logits(pre_label, label)
        loss = loss_1.mean() + loss2
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), n=len(torch.unique(data.batch)))
    return loss_meter.avg


def test(test_loader, model, args):
    global device
    model.eval()  # switch to test mode
    loss_meter = AverageMeter()
    acc_total = 0.0
    for data in test_loader:
        #print(data.name)
        data = data.to(device)
        with torch.no_grad():
            pre_label, label = model(data)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pre_label, label.float())
            loss_meter.update(loss.item(), n=len(torch.unique(data.batch)))
            accumulate_start_id = 0
            for i in range(len(torch.unique(data.batch))):
                pred_root_id = torch.argmax(pre_label[accumulate_start_id:accumulate_start_id + data.num_joint[i]]).item()
                gt_root_id = torch.argmax(label[accumulate_start_id:accumulate_start_id + data.num_joint[i]]).item()
                if pred_root_id == gt_root_id:
                    acc_total += 1.0
                accumulate_start_id += data.num_joint[i]
    return loss_meter.avg, acc_total/loss_meter.count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='network for picking root')
    parser.add_argument('--arch', default='rootnet')  # rootnet_fc, rootnet_nogt, rootnet_nogs
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200], help='Decrease learning rate at these epochs.')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    ####################################################################################################################
    parser.add_argument('--train-batch', default=3, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test-batch', default=3, type=int, metavar='N', help='test batchsize')
    parser.add_argument('-c', '--checkpoint', default='checkpoints/test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--logdir', default='logs/test', type=str, metavar='LOG', help='directory to save logs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_folder', default='/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/train/',
                        type=str, help='folder of training data')
    parser.add_argument('--val_folder', default='/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/val/',
                        type=str, help='folder of validation data')
    parser.add_argument('--test_folder', default='/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/test/',
                        type=str, help='folder of testing data')
    parser.add_argument('--pos_weight', default=10.0, type=float, help='weight for positive class')
    parser.add_argument('--topk', default=0.3, type=float, help='topk ratio for ohem')
    print(parser.parse_args())
    main(parser.parse_args())
