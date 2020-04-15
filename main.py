# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

import torch
from apex import amp
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils
import Read_data
from model.GNet import GNet
from save_graphs import save_graph

# torch.autograd.set_detect_anomaly(True)


def main():
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='lr scheduler patience (default: 10)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--name', type=str, default='Prueba',
                        help='Name of the current test (default: Prueba)')

    parser.add_argument('--load_model', type=str, default='best_acc',
                        help='Weights to load (default: best_acc)')
    parser.add_argument('--test', action='store_false', default=True,
                        help='Only test the model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Continue training a model')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Name of the folder with the pretrained model')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='Fine-tune a model')

    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU(s) to use (default: 0)')
    args = parser.parse_args()

    training = args.test
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.ft:
        args.resume = True

    args.patch_size = [128, 128, 96]
    args.num_classes = 2

    # PATHS AND DIRS
    save_path = os.path.join('TRAIN', args.name)
    out_path = os.path.join(save_path, 'Val')
    load_path = save_path
    if args.load_path is not None:
        load_path = os.path.join('TRAIN/', args.load_path)

    root = 'Heart'
    train_file = 'train_paths.csv'
    test_file = 'val_paths.csv'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(out_path)

    # SEEDS
    np.random.seed(12345)
    torch.manual_seed(12345)

    cudnn.deterministic = False
    cudnn.benchmark = True

    # CREATE THE NETWORK ARCHITECTURE
    model = GNet(num_classes=args.num_classes, backbone='xception')
    print('---> Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-5, amsgrad=True)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    annealing = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=args.patience, threshold=0.001,
        factor=0.5, threshold_mode="abs")

    criterion = utils.segmentation_loss(alpha=1)
    metrics = utils.Evaluator(args.num_classes)

    # LOAD A MODEL IF NEEDED (TESTING OR CONTINUE TRAINING)
    args.epoch = 0
    best_acc = 0
    if args.resume or not training:
        name = 'epoch_' + args.load_model + '.pth.tar'
        checkpoint = torch.load(
            os.path.join(load_path, name),
            map_location=lambda storage, loc: storage)
        args.epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        args.lr = checkpoint['lr']

        print('Loading model and optimizer {}.'.format(args.epoch))

        amp.load_state_dict(checkpoint['amp'])
        model.load_state_dict(checkpoint['state_dict'], strict=(not args.ft))
        if not args.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])

    # DATALOADERS
    train_loader = Read_data.MRIdataset(True, train_file, root,
                                        args.patch_size)
    val_loader = Read_data.MRIdataset(True, test_file, root, args.patch_size,
                                      val=True)
    test_loader = Read_data.MRIdataset(False, test_file, root, args.patch_size)

    train_loader = DataLoader(train_loader, shuffle=True, sampler=None,
                              batch_size=args.batch, num_workers=10)
    val_loader = DataLoader(val_loader, shuffle=False, sampler=None,
                            batch_size=args.batch * 2, num_workers=10)
    test_loader = DataLoader(test_loader, shuffle=False, sampler=None,
                             batch_size=1, num_workers=0)

    # TRAIN THE MODEL
    is_best = True
    if training:
        torch.cuda.empty_cache()
        out_file = open(os.path.join(save_path, 'progress.csv'), 'a+')

        for epoch in range(args.epoch + 1, args.epochs + 1):
            args.epoch = epoch
            lr = utils.get_lr(optimizer)
            print('--------- Starting Epoch {} --> {} ---------'.format(
                epoch, time.strftime("%H:%M:%S")))
            print('Learning rate:', lr)

            train_loss = train(args, model, train_loader, optimizer, criterion)
            val_loss, acc = val(args, model, val_loader, criterion, metrics)

            acc = acc.item()
            out_file.write('{},{},{},{},{}\n'.format(
                args.epoch, train_loss, val_loss, acc))
            out_file.flush()

            annealing.step(val_loss)
            save_graph(save_path)

            is_best = best_acc < acc
            best_acc = max(best_acc, acc)

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'loss': [train_loss, val_loss],
                'lr': lr,
                'acc': acc,
                'best_acc': best_acc}

            checkpoint = epoch % 20 == 0
            utils.save_epoch(state, save_path, epoch,
                             checkpoint=checkpoint, is_best=is_best)

            if lr <= (args.lr / (10 ** 3)):
                print('Stopping training: learning rate is too small')
                break
        out_file.close()

    # TEST THE MODEL
    if not is_best:
        checkpoint = torch.load(
            os.path.join(save_path, 'epoch_best_acc.pth.tar'),
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        args.epoch = checkpoint['epoch']
        print('Testing epoch with best dice ({}: acc {})'.format(
            args.epoch, checkpoint['acc']))

    test(args, model, test_loader, out_path, test_file)


def train(args, model, loader, optimizer, criterion):
    model.train()
    loader.dataset.change_epoch()
    epoch_loss = utils.AverageMeter()
    batch_loss = utils.AverageMeter()

    print_stats = len(loader) // 2
    for batch_idx, sample in enumerate(loader):
        data = Variable(sample['data'].float()).cuda()
        target = Variable(sample['target'].long()).cuda()

        out = model(data)
        loss = criterion(out, target)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if batch_loss.count % print_stats == 0:
            text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(text.format(
                time.strftime("%H:%M:%S"), (batch_idx + 1),
                (len(loader)), 100. * (batch_idx + 1) / (len(loader)),
                batch_loss.avg))
            batch_loss.reset()
    print('--- Train: \tLoss: {:.6f} ---'.format(epoch_loss.avg))
    return epoch_loss.avg


def val(args, model, loader, criterion, metrics):
    model.eval()
    metrics.reset()
    epoch_loss = utils.AverageMeter()

    for batch_idx, sample in enumerate(loader):
        data = Variable(sample['data'].float()).cuda()
        target = Variable(sample['target'].long()).cuda()

        with torch.no_grad():
            out = model(data)
        loss = criterion(out, target)

        prediction = F.softmax(out, dim=1)
        prediction = torch.argmax(prediction, dim=1).cpu().numpy()
        metrics.add_batch(target.cpu().numpy(), prediction)

        epoch_loss.update(loss.item(), n=target.shape[0])
    dice = metrics.Dice_Score()
    print('--- Val: \tLoss: {:.6f} \tDice fg: {} ---'.format(
        epoch_loss.avg, dice))
    return epoch_loss.avg, dice


def test(args, model, loader, save_path, test_file):
    patients = pd.read_csv('Paths/' + test_file)
    model.eval()

    # Add more weight to the central voxels
    w_patch = torch.zeros(args.patch_size)
    center = torch.Tensor(args.patch_size) // 2
    sigmas = torch.Tensor(args.patch_size) // 8
    w_patch[tuple(center.long())] = 1
    w_patch = gaussian_filter(w_patch, sigmas, 0, mode='constant', cval=0)
    w_patch = w_patch / w_patch.max()

    for idx in range(len(patients)):
        shape, name, affine = loader.dataset.update(idx)
        prediction = np.zeros((args.num_classes,) + shape)
        weights = np.zeros(shape)

        for sample in loader:
            data = Variable(sample['data'].float())
            voxel = Variable(sample['target'].squeeze_())
            low = (voxel - center).long()
            up = (voxel + center).long()
            with torch.no_grad():
                output = model(data.cuda(), False).cpu().numpy()[0]
            output *= w_patch
            prediction[:, low[0]:up[0], low[1]:up[1], low[2]:up[2]] += output
            weights[low[0]:up[0], low[1]:up[1], low[2]:up[2]] += w_patch

        prediction /= weights
        prediction = F.softmax(torch.Tensor(prediction), dim=0)
        prediction = torch.argmax(prediction, dim=0)

        Read_data.save_image(prediction, os.path.join(save_path, name), affine)
        print('Prediction {} saved'.format(name))


if __name__ == '__main__':
    main()
