# -*- coding: utf-8 -*-
import os
import time
import torch
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset


class MRIdataset(Dataset):
    def __init__(self, train, csv_file, root_dir, patch_size, val=False):
        super(MRIdataset, self).__init__()
        self.filenames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.train = train
        self.val = val
        self.patch_size = np.asarray(patch_size)
        self.fg = 0

    def __len__(self):
        if self.train:
            return len(self.filenames)
        else:
            return len(self.voxels)

    def __getitem__(self, idx):
        if self.train:
            patient = self.filenames.iloc[int(idx)]
            image, label, _ = load_image(patient, self.root_dir, self.train)
            # Just in case one of the axis is smaller than it should
            if image.shape[-1] <= self.patch_size[-1]:
                dif = self.patch_size[-1] - image.shape[-1] + 1
                pad = tuple(zip((0, 0, dif), (0, 0, dif)))
                image = np.pad(image, pad, 'reflect')
                label = np.pad(label, pad, 'reflect')

            if self.val:
                voxels = np.asarray(label.shape) // 2
            else:
                fg = (idx + self.fg) % 2 == 0
                voxels = train_voxels(image, self.patch_size, label, fg)

            # Patch extraction
            patches, label = make_batch(voxels, image, self.patch_size,
                                        self.train, label=label)
        else:
            patches, label = make_batch(self.voxels[idx], self.image,
                                        self.patch_size, self.train)
            patches = torch.from_numpy(patches)
            label = torch.Tensor(self.voxels[idx])

        return {'data': patches, 'target': label}

    def change_epoch(self):
        self.fg = 1 - self.fg

    def update(self, im_idx):
        # This is only for testing
        patient = self.filenames.iloc[im_idx]
        index = patient[0].find('r/') + 2
        name = patient[0][index:]
        print('Loading data of patient {} ---> {}'.format(
            name, time.strftime("%H:%M:%S")))

        self.image, _, affine = load_image(patient, self.root_dir, self.train)
        im_shape = self.image.shape

        self.voxels = test_voxels(self.patch_size, im_shape)
        return im_shape, name, affine


def list_split(lista, parts):
    size = len(lista) // parts
    output = [lista[i:i + size] for i in range(0, len(lista), size)]
    return output


def test_voxels(patch_size, im_shape):
    center = patch_size // 2
    dims = []
    for i, j in zip(im_shape, center):
        end = i - j
        num = np.ceil((end - j) / j)
        if num == 1:
            num += 1
        voxels = np.linspace(j, end, int(num))
        dims.append(voxels)
    voxels = list(itertools.product(*dims))
    return voxels


def train_voxels(image, patch_size, label, foreground):
    low = patch_size // 2
    high = np.asarray(image.shape) - low

    if foreground:
        # Force the center voxel to belong to a foreground category
        pad = tuple(zip(low, low))
        mask = np.pad(np.zeros(high - low), pad, 'constant',
                      constant_values=-1)

        np.copyto(mask, label, where=(mask == 0))
        fg = np.unique(mask)[2:]  # [ignore, bg, fg...]
        cat = np.random.choice(fg)
        selected = np.argwhere(mask == cat)
        coords = selected[np.random.choice(len(selected))]
    else:
        x = np.random.randint(low[0], high[0])
        y = np.random.randint(low[1], high[1])
        z = np.random.randint(low[2], high[2])
        coords = (x, y, z)
    return coords


def load_image(patient, root_dir, train):
    image = []
    gt = None
    for modality in patient:
        im_path = os.path.join(root_dir, modality)
        im = nib.load(im_path)
        if 'labelsTr' in modality:
            gt = im.get_data().astype(np.int16)
            continue
        image.append(im.get_data())
    return np.squeeze(image[0]), gt, im.affine


def make_batch(voxel, image, patch_size, train, label=[]):
    patch = extract_patch(image, voxel, patch_size)
    gt = np.zeros(1)
    if train:
        gt = extract_patch(label, voxel, patch_size)
    return patch, gt


def extract_patch(image, voxel, patch_size):
    im_size = image.shape
    v1 = np.asarray(voxel) - patch_size // 2
    v1 = v1.astype(int)
    v2 = np.minimum(v1 + patch_size, im_size)

    patch_list = []
    patch = image[v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]]
    patch = verify_size(patch, patch_size)
    patch_list.append(patch)
    return np.stack(patch_list, axis=0)


def verify_size(im, size):
    dif = np.asarray(size) - im.shape
    if any(dif > 0):
        dif[dif < 0] = 0
        mod = dif % 2
        dif = np.abs(dif) // 2
        pad = tuple(zip(dif, dif + mod))
        im = np.pad(im, pad, 'reflect')
    return im


def save_image(prediction, outpath, affine):
    new_pred = nib.Nifti1Image(prediction.numpy(), affine)
    new_pred.set_data_dtype(np.uint8)
    nib.save(new_pred, outpath)
