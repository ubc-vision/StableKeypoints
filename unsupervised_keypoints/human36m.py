import os
import h5py
from tqdm import tqdm
import numpy as np
import scipy.io
import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from matplotlib import colors


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

        self.data_root = data_root

        self.samples = []

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        mask = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'BackgroudMask',
                                       folder_names, '{}.png'.format(frame_index)))

        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        mask_array = torch.from_numpy(np.array(mask))

        # Resize the mask to [1, 512, 512, 3]
        resized_mask_array = F.interpolate(mask_array[None, None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]

        if img_array.shape[-1] != 512:
            img_array = F.interpolate(img_array[None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]

        # Element-wise multiplication
        result_img = img_array * resized_mask_array
        # result_img = img_array

        return {'img': result_img}

    def __len__(self):
        return len(self.samples)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()

        self.data_root = data_root

        self.samples = []

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        mask = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'BackgroudMask',
                                       folder_names, '{}.png'.format(frame_index)))
        keypoints = scipy.io.loadmat(os.path.join(self.data_root, 'S{}'.format(subject_index), 'Landmarks',
                                                  folder_names, '{}.mat'.format(frame_index)))['keypoints_2d'].astype(np.float32)

        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        mask_array = torch.from_numpy(np.array(mask))

        # Resize the mask to [1, 512, 512, 3]
        resized_mask_array = F.interpolate(mask_array[None, None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]
        
        if img_array.shape[-1] != 512:
            img_array = F.interpolate(img_array[None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]

        # Element-wise multiplication
        result_img = img_array * resized_mask_array
        # result_img = img_array

        return {'img': result_img, 'kpts': torch.tensor(keypoints), 'visibility': torch.ones(keypoints.shape[0])}

    def __len__(self):
        return len(self.samples)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()

        self.data_root = data_root

        self.samples = []

        for subject_index in [11]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        mask = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index), 'BackgroudMask',
                                       folder_names, '{}.png'.format(frame_index)))
        keypoints = scipy.io.loadmat(os.path.join(self.data_root, 'S{}'.format(subject_index), 'Landmarks',
                                                  folder_names, '{}.mat'.format(frame_index)))['keypoints_2d'].astype(np.float32)

        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        mask_array = torch.from_numpy(np.array(mask))

        # Resize the mask to [1, 512, 512, 3]
        resized_mask_array = F.interpolate(mask_array[None, None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]
        
        if img_array.shape[-1] != 512:
            img_array = F.interpolate(img_array[None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]

        # Element-wise multiplication
        result_img = img_array * resized_mask_array
        # result_img = img_array

        return {'img': result_img, 'kpts': torch.tensor(keypoints), 'visibility': torch.ones(keypoints.shape[0])}

    def __len__(self):
        return len(self.samples)
    
    
if __name__ == "__main__":
    
    # Initialize your dataset
    data_root = '/home/iamerich/burst/human_images_og/human_images'
    image_size = 512  # for example
    dataset = TrainSet(data_root, image_size)
    
    # import ipdb; ipdb.set_trace()

    # Prepare to convert the dataset
    # Open an HDF5 file in 'w'rite mode
    with h5py.File('/home/iamerich/scratch/human36m/TrainSet.h5', 'w') as h5f:
        # Preallocate space for the largest expected size in your dataset
        # This assumes all images are the same size and the poses have the same shape
        images_dataset = h5f.create_dataset('images', (len(dataset), 3, image_size, image_size), dtype='f')
        # poses_dataset = h5f.create_dataset('kpts', (len(dataset), 32, 2), dtype='f')
        # visibility_dataset = h5f.create_dataset('visibility', (len(dataset), 32), dtype='i')

        # Convert each item and save it to the HDF5 file
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            images_dataset[i] = sample['img']
            # poses_dataset[i] = sample['kpts']
            # visibility_dataset[i] = sample['visibility']

        # Optionally you can also save some attributes or metadata if needed
        h5f.attrs['image_size'] = image_size