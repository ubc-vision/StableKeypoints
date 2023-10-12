import os

import numpy as np
import scipy.io
import torch
import torch.utils.data
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
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.to_tensor = transforms.ToTensor()

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
        
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        # Expand dimensions of mask_array to (128, 128, 3) by repeating it across the third dimension
        expanded_mask_array = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2)

        # Element-wise multiplication
        result_array = img_array * expanded_mask_array

        # Convert the result back to an image
        result_img = Image.fromarray(result_array.astype('uint8'))
        

        return {'img': self.transform(result_img)}

    def __len__(self):
        return len(self.samples)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.to_tensor = transforms.ToTensor()

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
        
        
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        # Expand dimensions of mask_array to (128, 128, 3) by repeating it across the third dimension
        expanded_mask_array = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2)

        # Element-wise multiplication
        result_array = img_array * expanded_mask_array

        # Convert the result back to an image
        result_img = Image.fromarray(result_array.astype('uint8'))
        
        # import matplotlib.pyplot as plt
        # part_color = get_part_color(keypoints.shape[0])
        # plt.imshow(result_img)
        # for i in range(keypoints.shape[0]):
        #     plt.scatter(keypoints[i, 1]*128, keypoints[i, 0]*128, c=part_color[i])
        #     plt.annotate(str(i), (keypoints[i, 1]*128, keypoints[i, 0]*128))
        # plt.savefig("outputs/taichi.png")

        return {'img': self.transform(result_img), 'kpts': torch.tensor(keypoints), 'visibility': torch.ones(keypoints.shape[0])}

    def __len__(self):
        return len(self.samples)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.to_tensor = transforms.ToTensor()

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
        
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        # Expand dimensions of mask_array to (128, 128, 3) by repeating it across the third dimension
        expanded_mask_array = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2)

        # Element-wise multiplication
        result_array = img_array * expanded_mask_array

        # Convert the result back to an image
        result_img = Image.fromarray(result_array.astype('uint8'))

        return {'img': self.transform(result_img), 'kpts': torch.tensor(keypoints), 'visibility': torch.ones(keypoints.shape[0])}

    def __len__(self):
        return len(self.samples)
    
    
if __name__ == "__main__":
    
    # select random number between 0 and 4999
    num = np.random.randint(0, 5000)
    print("Testing taichi.py")
    dataset = TrainRegSet(data_root="/ubc/cs/home/i/iamerich/scratch/datasets/human3.6m/human_images", image_size=512)
    
    print("len(dataset):", len(dataset))
    
    batch = dataset[num]
    
    print("batch['img'].shape:", batch['img'].shape)
    
    pass