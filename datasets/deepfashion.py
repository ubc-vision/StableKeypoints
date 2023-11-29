"""
Code adapted from: https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints/blob/main/datasets/deepfashion.py
MIT License

Copyright (c) 2023 xingzhehe
"""

import json
import os

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.img_file = [l.split(',')[1].strip() for l in open(os.path.join(data_root, 'data_train.csv'))][1:]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'img', self.img_file[idx]))
        sample = {'img': self.transform(img)}
        return sample

    def __len__(self):
        return len(self.img_file)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.img_file = [l.split(',')[1].strip() for l in open(os.path.join(data_root, 'data_train.csv'))][1:]

        with open(os.path.join(data_root, 'data_train.json'), 'r') as f:
            self.keypoints = json.load(f)
        self.keypoints = [self.keypoints[i]['keypoints'] for i in range(len(self.keypoints))]
        self.keypoints = torch.tensor(self.keypoints).roll(shifts=1, dims=-1)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'img', self.img_file[idx]))
        sample = {'img': self.transform(img), 'kpts': self.keypoints[idx]/256}
        return sample

    def __len__(self):
        return len(self.img_file)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.data_root = data_root
        self.img_file = [l.split(',')[1].strip() for l in open(os.path.join(data_root, 'data_test.csv'))][1:]

        with open(os.path.join(data_root, 'data_test.json'), 'r') as f:
            self.keypoints = json.load(f)
        self.keypoints = [self.keypoints[i]['keypoints'] for i in range(len(self.keypoints))]
        self.keypoints = torch.tensor(self.keypoints).roll(shifts=1, dims=-1)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, 'img', self.img_file[idx]))
        sample = {'img': self.transform(img), 'kpts': self.keypoints[idx]/256}
        return sample

    def __len__(self):
        return len(self.img_file)