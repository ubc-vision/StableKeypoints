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
        
        
        # import matplotlib.pyplot as plt
        # part_color = get_part_color(18)
        # plt.imshow(img.permute(1, 2, 0))
        # for i in range(18):
        #     plt.scatter(pose[i, 1]*512, pose[i, 0]*512, c=part_color[i])
        #     plt.annotate(str(i), (pose[i, 1]*512, pose[i, 0]*512))
        # plt.savefig("outputs/taichi.png")
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

        # for i in range(len(pose_file)):
        #     image_file = pose_file.file_name[i]
        #     img = Image.open(os.path.join(data_root, 'eval_images', 'taichi-256', 'test', image_file))
        #     img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        #     seg = Image.open(os.path.join(data_root, 'taichi-test-masks', image_file))
        #     seg = seg.resize((image_size, image_size), resample=Image.BILINEAR)
        #     self.imgs.append(np.asarray(img) / 255)
        #     self.segs.append(np.asarray(seg) / 255)
        #     self.poses.append(pose_file.value[i])  # [0, 255]

        # self.imgs = torch.tensor(np.array(self.imgs)).float().permute(0, 3, 1, 2)
        # self.imgs = self.imgs.contiguous()
        # self.segs = torch.tensor(self.segs).int()
        # self.poses = torch.tensor(self.poses).float()
        # self.poses = torch.cat([self.poses[:, :, 1:2], self.poses[:, :, 0:1]], dim=2)

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
        
        
        # import matplotlib.pyplot as plt
        # part_color = get_part_color(18)
        # plt.imshow(img.permute(1, 2, 0))
        # for i in range(18):
        #     plt.scatter(pose[i, 1]*512, pose[i, 0]*512, c=part_color[i])
        #     plt.annotate(str(i), (pose[i, 1]*512, pose[i, 0]*512))
        # plt.savefig("outputs/taichi.png")
        sample = {'img': img, 'kpts': pose, 'visibility': visibility}
        return sample
    


    def __len__(self):
        return len(self.pose_file)


def regress_kp(batch_list):
    train_X = torch.cat([batch['det_keypoints'] for batch in batch_list]) * 0.5 + 0.5
    train_X = train_X * 255
    train_y = torch.cat([batch['keypoints'] for batch in batch_list])
    scores = []
    num_gnd_kp = 18
    betas = []
    for i in range(num_gnd_kp):
        for j in range(2):
            index = (train_y[:, i, j] + 1).abs() > 1e-6
            features = train_X[index]
            features = features.reshape(features.shape[0], -1)
            label = train_y[index, i, j]
            features = torch.cat([features, torch.ones_like(features[:, -1:])], dim=1)
            try:
                beta = (features.T @ features).inverse() @ features.T @ label
            except:
                beta = (features.T @ features + torch.eye(features.shape[-1]).to(features)).inverse() @ features.T @ label
            betas.append(beta)

            pred_label = features @ beta
            score = (pred_label - label).abs().mean()
            scores.append(score.item())
    return {'val_loss': np.sum(scores), 'beta': betas}


def test_epoch_end(batch_list_list):
    valid_list = batch_list_list[0]
    test_list = batch_list_list[1]
    betas = regress_kp(valid_list)['beta']
    num_gnd_kp = 18
    scores = []

    X = torch.cat([batch['det_keypoints'] for batch in test_list]) * 0.5 + 0.5
    X = X * 255
    y = torch.cat([batch['keypoints'] for batch in test_list])

    beta_index = 0

    for i in range(num_gnd_kp):
        for j in range(2):
            index_test = (y[:, i, j] + 1).abs() > 1e-6
            features = X[index_test]
            features = features.reshape(features.shape[0], -1)
            features = torch.cat([features, torch.ones_like(features[:, -1:])], dim=1)
            label = y[index_test, i, j]
            pred_label = features @ betas[beta_index]
            score = (pred_label - label).abs().mean()
            scores.append(score.item())
            beta_index += 1

    return {'val_loss': np.sum(scores)}


if __name__ == "__main__":
    
    # Initialize your dataset
    data_root = '/home/iamerich/burst/taichi/'
    image_size = 512  # for example
    dataset = TrainSet(data_root, image_size)
    
    # import ipdb; ipdb.set_trace()

    # Prepare to convert the dataset
    # Open an HDF5 file in 'w'rite mode
    with h5py.File('/home/iamerich/scratch/taichi/TrainSet.h5', 'w') as h5f:
        # Preallocate space for the largest expected size in your dataset
        # This assumes all images are the same size and the poses have the same shape
        images_dataset = h5f.create_dataset('images', (len(dataset), 3, image_size, image_size), dtype='f')
        # poses_dataset = h5f.create_dataset('kpts', (len(dataset), dataset.pose_file.value[0].shape[0], 2), dtype='f')
        # visibility_dataset = h5f.create_dataset('visibility', (len(dataset), dataset.pose_file.value[0].shape[0]), dtype='i')

        # Convert each item and save it to the HDF5 file
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            images_dataset[i] = sample['img']
            # poses_dataset[i] = sample['kpts']
            # visibility_dataset[i] = sample['visibility']

        # Optionally you can also save some attributes or metadata if needed
        h5f.attrs['image_size'] = image_size