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
        # import matplotlib.pyplot as plt
        # part_color = get_part_color(15)
        # plt.imshow(self.imgs[idx].permute(1, 2, 0))
        # for i in range(15):
        #     if self.visibility[idx, i] == 1:
        #         plt.scatter(self.keypoints[idx, i, 1]*128, self.keypoints[idx, i, 0]*128, c=part_color[i])
        #         plt.annotate(str(i), (self.keypoints[idx, i, 1]*128, self.keypoints[idx, i, 0]*128))
        # plt.show()
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
        sample = {'img': self.transform(self.imgs[idx] / 255), 'kpts': self.keypoints[idx], 'visibility': self.visibility[idx]}
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
        sample = {'img': self.transform(self.imgs[idx] / 255), 'kpts': self.keypoints[idx], 'visibility': self.visibility[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


def regress_kp(batch_list):
    train_X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    train_y = torch.cat([batch['kpts'] for batch in batch_list])
    visibility = torch.cat([batch['visibility'] for batch in batch_list])
    scores = []
    num_gnd_kp = 15
    betas = []
    for i in range(num_gnd_kp):
        index = visibility[:, i].bool()
        if index.sum() == 0:
            betas.append(torch.zeros(2*train_X.shape[1], 2))
            continue
        features = train_X[index]
        features = features.reshape(features.shape[0], -1)
        label = train_y[index, i]
        try:
            beta = (features.T @ features).inverse() @ features.T @ label
        except:
            beta = (features.T @ features + torch.eye(features.shape[-1]).to(features)).inverse() @ features.T @ label
        betas.append(beta)

        pred_label = features @ beta
        score = (pred_label - label).norm(dim=-1).sum()
        scores.append(score.item())
    return {'val_loss': np.sum(scores) / visibility.sum().item(), 'beta': betas}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    betas = regress_kp(valid_list)['beta']
    num_gnd_kp = 15
    scores = []

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    y = torch.cat([batch['kpts'] for batch in test_list])
    visibility = torch.cat([batch['visibility'] for batch in test_list])

    for i in range(num_gnd_kp):
        index_test = visibility[:, i].bool()
        if index_test.sum() == 0:
            continue
        features = X[index_test]
        features = features.reshape(features.shape[0], -1)
        label = y[index_test, i]
        pred_label = features @ betas[i]
        score = (pred_label - label).norm(dim=-1).sum()
        scores.append(score.item())

    return {'val_loss': np.sum(scores) / visibility.sum().item()}