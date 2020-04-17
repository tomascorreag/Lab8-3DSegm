# -*- coding: utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_epoch(state, save_path, epoch, checkpoint, is_best):
    if checkpoint:
        name = 'epoch_' + str(epoch) + '.pth.tar'
        torch.save(state, os.path.join(save_path, name))
        print('Checkpoint saved:', name)

    if is_best:
        name = 'epoch_best_acc.pth.tar'
        torch.save(state, os.path.join(save_path, name))
        print('New best model saved')

    name = 'epoch_last.pth.tar'
    torch.save(state, os.path.join(save_path, name))


def one_hot(gt, categories):
    size = [*gt.shape] + [categories]
    y = gt.view(-1, 1)
    gt = torch.FloatTensor(y.nelement(), categories).zero_().cuda()
    gt.scatter_(1, y, 1)
    gt = gt.view(size).permute(0, 4, 1, 2, 3)
    return gt


# ================= Keep stats =================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.conf = np.zeros((self.num_class,) * 2)  # Confusion matrix

    def Pixel_Accuracy(self):
        Acc = np.diag(self.conf).sum() / self.conf.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.conf) / self.conf.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.conf) / (
                    np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0) -
                    np.diag(self.conf))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.conf, axis=1) / np.sum(self.conf)
        iu = np.diag(self.conf) / (
                    np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0) -
                    np.diag(self.conf))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Dice_Score(self):
        MDice = (2 * np.diag(self.conf)) / (
                    np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0))
        return MDice[1:]  # Only foreground

    def Frequency_Weighted_Dice_Score(self):
        freq = np.sum(self.conf, axis=1) / np.sum(self.conf)
        dice = (2 * np.diag(self.conf)) / (
                    np.sum(self.conf, axis=1) + np.sum(self.conf, axis=0))

        FWDice = (freq[freq > 0] * dice[freq > 0]).sum()
        return FWDice

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        conf = count.reshape(self.num_class, self.num_class)
        return conf

    def add_batch(self, pre_image, gt_image):
        assert gt_image.shape == pre_image.shape
        self.conf += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.conf = np.zeros((self.num_class,) * 2)


# ================= LOSSES =================
class tversky_loss(nn.Module):
    """
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
    """
    def __init__(self, alpha, smooth=1):
        super(tversky_loss, self).__init__()
        self.alpha = alpha
        self.beta = 2 - alpha
        self.smooth = smooth

    def forward(self, inputs, targets):
        targets = one_hot(targets, inputs.shape[1])
        inputs = F.softmax(inputs, dim=1)

        dims = tuple(range(2, targets.ndimension()))
        tps = torch.sum(inputs * targets, dims)
        fps = torch.sum(inputs * (1 - targets), dims) * self.alpha
        fns = torch.sum((1 - inputs) * targets, dims) * self.beta
        loss = (2 * tps + self.smooth) / (2 * tps + fps + fns + self.smooth)
        loss = torch.mean(loss, dim=0)
        return 1 - (loss[1:]).mean()


class segmentation_loss(nn.Module):
    def __init__(self, alpha):
        super(segmentation_loss, self).__init__()
        self.dice = tversky_loss(alpha=alpha, smooth=1)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets.contiguous())
        ce = self.ce(inputs, targets)
        return dice + ce
