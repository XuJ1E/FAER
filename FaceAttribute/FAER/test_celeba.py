# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2024/1/11 22:46
# Author     ：XuJ1E
# version    ：python 3.8
# File       : test_celeba.py
"""
import time
import sys
import argparse
import numpy as np
sys.path.append('.')
import yaml
import torch
import torch.utils.data
from loguru import logger

from utils.utli import *
from utils.loss import Loss
from sklearn.metrics import multilabel_confusion_matrix
from celeba_dataset import MyDataset
from torchvision import transforms
from models.convnext import convnext_base

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path', default=r'./dataset/CelebA/', help='path to dataset')
parser.add_argument('--eval_only', default=True, help='store_true')
parser.add_argument('--eval_ckpt', default="./pretrain/model_best.pth",
                    type=str, help='path to checkpoint for evaluation')
parser.add_argument('--resume', default=None, type=str, help='path to checkpoint for resume')
parser.add_argument('--model_path', default='./weight/model', help='path for model checkpoint')
parser.add_argument('--num_classes', default=40, type=int, help='num_classes for prediction')
parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=35, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--arch', default='FAER_CELEBA', type=str, help='model type for train')
parser.add_argument('--ratio', default=1, type=float, help='mask ratio for BEC loss function')
parser.add_argument('--drop_path', default=0.25, type=float, help='drop layer of model')
parser.add_argument('--lr', default=2.5e-4, type=float, help='initial learning rate')
parser.add_argument('--lr_new', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--decay-steps', default=None, type=int, help='decay step for learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=0.001, type=float, help='weight decay (default: 1e-4)', )
parser.add_argument('--print-freq', default=100, type=int, help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=102, type=int)


def main(args):

    exp_path = os.path.join('experiment/', time.strftime("%m_%d_%H_%M_%S", time.localtime()))
    os.makedirs(exp_path, exist_ok=True)
    logger.add(os.path.join(exp_path, 'log.txt'))

    seedForExp(args)
    logger.info('Epochs:', args.epochs, 'seed:', args.seed, 'Bs:', args.batch_size)
    logger.info("Use GPU: {} for training".format(args.gpu))
    logger.info(args)

    best_err = 100
    model = convnext_base(pretrained=True, num_classes=args.num_classes, drop_path_rate=args.drop_path)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    train_dataset = MyDataset(root=args.data_path,
                              mode='train',
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(15),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    test_dataset = MyDataset(root=args.data_path,
                             mode='test',
                             transform=transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    criterion = Loss(mask_ratio=args.ratio).cuda()

    optimizer = torch.optim.AdamW(params=[{'params': model.module.base_parameters(), 'lr': args.lr / 10},
                                          {'params': model.module.large_parameters()}], lr=args.lr)

    if args.eval_only:
        model.eval()
        logger.info(f'Load model from {args.eval_ckpt}')
        model.load_state_dict(torch.load(args.eval_ckpt)['state_dict'])
        eval_one_epoch(test_loader, model, criterion, 0)
        exit()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_acc']
            best_err = best_err.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args, args.lr)
        t_err, t_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        e_err, e_loss, = eval_one_epoch(test_loader, model, criterion, 0)

        is_best = e_err < best_err
        best_err = min(e_err, best_err)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_err,
            'optimizer': optimizer.state_dict()
        }, is_best, args)
        print(f'>>>>>>Best err [{best_err}]')

    logger.info(best_err)
    with open(os.path.join(exp_path, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(vars(args), f)
    os.rename(exp_path, exp_path + '_%.2f' % best_err)
    return best_err


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    err = AverageMeter('Error', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, err],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True).float()
        _, pred = model(images)
        loss = criterion(pred, target)
        if loss != loss:
            logger.warning('nan loss')
            sys.exit()
        res = torch.where(pred > 0, torch.ones_like(pred), torch.zeros_like(pred))
        err_batch = 100 - torch.sum(res == target) * (100 / (target.shape[0] * target.shape[1]))

        err.update(err_batch, target.shape[0] * target.shape[1])
        batch_time.update(time.time() - end)
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            progress.display(i)

    return err.avg, losses.avg


def eval_one_epoch(test_loader, model, criterion, threshold: int):
    TASKS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
             'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
             'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
             'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
             'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    predict_label_list = [[] for _ in range(len(TASKS))]
    true_label_list = [[] for _ in range(len(TASKS))]
    highest_acc_list = [0.0 for _ in range(len(TASKS))]
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    err = AverageMeter('Error', ':6.2f')
    acc_metrics = [AverageMeter(f'Acc:{TASKS[j]}', ':6.2f') for j in range(len(TASKS))]
    stat_target = None
    err_spe = None
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True).float()
            _, pred = model(images)
            for j in range(len(TASKS)):
                spec_target = target[:, j].cuda()
                res_j = torch.where(pred[:, j] > threshold, torch.ones_like(pred[:, j]), torch.zeros_like(pred[:, j]))
                predict_label_list[j] += torch.sum(res_j == spec_target)
                true_label_list[j] += 1
                spec_acc = torch.sum(res_j == spec_target)*(100/spec_target.shape[0])
                acc_metrics[j].update(spec_acc, spec_acc.size[0])
            loss = criterion(pred, target)
            res = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred))
            stat_target = torch.sum(target.clamp(min=0), dim=0) if stat_target is None else stat_target + torch.sum(target.clamp(min=0), dim=0)
            err_spe = torch.sum(res == target, dim=0).int() if err_spe is None else err_spe + torch.sum(res == target, dim=0).int()
            err_batch = 100 - torch.sum(res == target) * (100 / (target.shape[0] * target.shape[1]))
            if loss != 0:
                losses.update(loss.item(), images.size(0))
            err.update(err_batch, target.shape[0] * target.shape[1])
            batch_time.update(time.time() - end)
            end = time.time()
        for j in range(len(TASKS)):
            if acc_metrics[j].avg > highest_acc_list[j]:
                highest_acc_list[j] = acc_metrics[j].avg
            msg = f'[Acc of {TASKS[j]}:] | {highest_acc_list[j]}'
            label_msg = f'[Label number of {TASKS[j]}:] | {predict_label_list[j]}'
            logger.info(msg)
            logger.info(label_msg)
        err_spe = 100 - err_spe.float() * (100 / len(test_loader.dataset))
        stat_target = stat_target * (100 / len(test_loader.dataset))
        msg = 'Test: Error@1 {err.avg:.3f} | Loss {losses.avg:.3f} | Time {batch_time.sum:.1f}'.format(err=err, losses=losses, batch_time=batch_time)
        logger.info(msg)
        return err.avg, losses.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
