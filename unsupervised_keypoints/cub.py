import os

import h5py
import numpy as np
import torch
import torch.utils.data
from matplotlib import colors
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
        data_file = 'cub.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])
            self.keypoints = torch.from_numpy(hf['train_kp'][...])  # [0, 1]
            self.visibility = torch.from_numpy(hf['train_vis'][...])  # 1 for visible and 0 for invisible

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255)}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'cub.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])
            self.keypoints = torch.from_numpy(hf['train_kp'][...])  # [0, 1]
            self.visibility = torch.from_numpy(hf['train_vis'][...])  # 1 for visible and 0 for invisible

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
        ])

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255), 'kpts': self.keypoints[idx][:, [1, 0]], 'visibility': self.visibility[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        data_file = 'cub.h5'
        with h5py.File(os.path.join(data_root, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['test_img'][...])
            self.keypoints = torch.from_numpy(hf['test_kp'][...])  # [0, 1]
            self.visibility = torch.from_numpy(hf['test_vis'][...])  # 1 for visible and 0 for invisible

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
        ])

    def __getitem__(self, idx):
        # swap x and y for keypoints
        # kpts = self.keypoints[idx][:, [1, 0]]
        sample = {'img': self.transform(self.imgs[idx] / 255), 'kpts': self.keypoints[idx][:, [1, 0]], 'visibility': self.visibility[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]
