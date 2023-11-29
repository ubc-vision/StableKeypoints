"""
Code adapted from: https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints/blob/main/datasets/taichi.py
MIT License

Copyright (c) 2023 xingzhehe
"""

import os
import h5py
import numpy as np
import pandas
import torch
import torch.utils.data
import torchvision
from PIL import Image
from matplotlib import colors
from tqdm import tqdm
from torchvision import transforms


def get_part_color(n_parts):
    colormap = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                'honeydew', 'thistle',
                'red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                'honeydew', 'thistle')[:n_parts]
    part_color = []
    for i in range(n_parts):
        part_color.append(colors.to_rgb(colormap[i]))
    part_color = np.array(part_color)

    return part_color

class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.imgs = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.image_size=image_size
        # self.imgs = []
        # self.poses = []

        with open(os.path.join(data_root, 'landmark', 'taichi_train_gt.pkl'), 'rb') as f:
            self.pose_file = pandas.read_pickle(f)

    def __getitem__(self, idx):
        
        image_file = self.pose_file.file_name[idx]
        img = Image.open(os.path.join(self.data_root, 'eval_images', 'taichi-256', 'train', image_file))
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        img = np.asarray(img) / 255
        img= torch.tensor(img).permute(2, 0, 1).float()
        pose = self.pose_file.value[idx]/256
        pose = torch.tensor(pose)
        # swap x and y
        pose = torch.cat([pose[:, 1:2], pose[:, 0:1]], dim=1)
        
        visibility = pose > 0
        visibility = (torch.sum(visibility, dim=1)==2)
        
        sample = {'img': img, 'kpts': pose, 'visibility': visibility}
        return sample

    def __len__(self):
        return len(self.pose_file)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size

        with open(os.path.join(data_root, 'landmark', 'taichi_test_gt.pkl'), 'rb') as f:
            self.pose_file = pandas.read_pickle(f)

    def __getitem__(self, idx):
        
        image_file = self.pose_file.file_name[idx]
        img = Image.open(os.path.join(self.data_root, 'eval_images', 'taichi-256', 'test', image_file))
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        img = np.asarray(img) / 255
        img= torch.tensor(img).permute(2, 0, 1).float()
        pose = self.pose_file.value[idx]/256
        pose = torch.tensor(pose)
        # swap x and y
        pose = torch.cat([pose[:, 1:2], pose[:, 0:1]], dim=1)
        
        visibility = pose > 0
        visibility = (torch.sum(visibility, dim=1)==2)
        
        sample = {'img': img, 'kpts': pose, 'visibility': visibility}
        return sample
    


    def __len__(self):
        return len(self.pose_file)