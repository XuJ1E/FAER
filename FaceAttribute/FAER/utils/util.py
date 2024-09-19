import shutil
import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from math import cos, pi
from loguru import logger


def adjust_learning_rate(optimizer, epoch, args, lr):
    global_step = epoch
    cosine_decay = 0.5 * (1 + cos(pi * global_step / args.decay_steps)) if args.decay_steps is not None else 0.5 * (
            1 + cos(pi * global_step / args.epochs))
    decayed = (1 - 1e-6) * cosine_decay + 1e-6
    decayed_learning_rate = lr * decayed
    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate
    return decayed_learning_rate


def seedForExp(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(torch.initial_seed())
    else:
        torch.manual_seed(1000000 * torch.rand(1))
        np.random.seed(torch.initial_seed())
        random.seed(torch.initial_seed())
        logger.info(torch.initial_seed())
        args.seed = torch.initial_seed()


def load_matched_state_dict(model, state_dict, flit=None, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()

    if flit is None:
        for key in curr_state_dict.keys():
            num_total += 1
            key_tofind = key

            if key_tofind in state_dict and curr_state_dict[key].shape == state_dict[key_tofind].shape:
                curr_state_dict[key] = state_dict[key_tofind]
                num_matched += 1
            else:
                logger.info(f'{key}, {curr_state_dict[key].shape}')

    else:
        for key in curr_state_dict.keys():
            num_total += 1
            key_tofind = flit + key
            if key_tofind in state_dict and curr_state_dict[key].shape == state_dict[key_tofind].shape:
                curr_state_dict[key] = state_dict[key_tofind]
                num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        logger.info(f'Loaded state_dict: {num_matched}/{num_total} matched')


def save_checkpoint(state, is_best, args, filename='_checkpoint.pth'):
    # return
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
    torch.save(state, args.model_path + '/' + args.arch + filename)
    logger.info(f'save to {args.model_path}/{args.arch}{filename}')
    if is_best:
        shutil.copyfile(args.model_path + '/' + args.arch + filename, args.model_path + '/' + args.arch + '_model_best.pth')


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
        logger.info('\t'.join(entries))

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


def voc_ap(rec, prec, true_num):
    # borrow from: https://github.com/HCPLab-SYSU/SSGRL/blob/master/utils/metrics.py
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(imagessetfilelist, num, return_each=False):
    # borrow from: https://github.com/HCPLab-SYSU/SSGRL/blob/master/utils/metrics.py
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())
    try:
        seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    except:
        breakpoint()
    gt_label = seg[:, num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims=True)

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_logger.infooptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP
