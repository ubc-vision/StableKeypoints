import os
import h5py
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
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index))):
                    if folder_names.startswith(action):
                        for camera in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), folder_names, "imageSequence")):
                            for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), folder_names, "imageSequence", camera)):
                                self.samples.append((subject_index, folder_names, camera, int(frame_index.split('.')[0].split("_")[1])))

    def __getitem__(self, idx):
        subject_index, folder_names, camera, frame_index = self.samples[idx]
        
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index),
                                      folder_names, "imageSequence", camera, f'img_{frame_index:06d}.jpg'))

        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255

        img_array = F.interpolate(img_array[None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]

        return {'img': img_array}

    def __len__(self):
        return len(self.samples)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()

        self.data_root = data_root

        self.samples = []

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index))):
                    if folder_names.startswith(action):
                        for camera in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), folder_names, "imageSequence")):
                            for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), folder_names, "imageSequence", camera)):
                                self.samples.append((subject_index, folder_names, camera, int(frame_index.split('.')[0].split("_")[1])))

    def __getitem__(self, idx):
        subject_index, folder_names, camera, frame_index = self.samples[idx]
        
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index),
                                      folder_names, "imageSequence", camera, f'img_{frame_index:06d}.jpg'))
        
        img_size = img.size

        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        
        img_array = F.interpolate(img_array[None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]

        annot_file = h5py.File(os.path.join(self.data_root, 'S{}'.format(subject_index), folder_names, "annot.h5"), "r")
        correct_cam = np.array(annot_file['camera'])==int(camera)
        correct_frame = np.array(annot_file['frame'])==frame_index
        assert (correct_cam*correct_frame).sum() == 1
        annot_frame = np.nonzero(correct_cam*correct_frame)[0][0]
        pose = annot_file['pose/2d'][annot_frame]
        pose /= img_size
        
        # transpose last 2 dimensions
        pose = pose[:, [1, 0]]
        
        # import matplotlib.pyplot as plt
        # plt.imshow(img_array.permute(1, 2, 0)) 
        # plt.scatter(pose[:, 1]*512, pose[:, 0]*512)
        # plt.savefig("temp.png")
        # pass

        return {'img': img_array, 'kpts': torch.tensor(pose), 'visibility': torch.ones(pose.shape[0])}

    def __len__(self):
        return len(self.samples)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()

        self.data_root = data_root

        self.samples = []

        for subject_index in [11]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index))):
                    if folder_names.startswith(action):
                        for camera in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), folder_names, "imageSequence")):
                            for frame_index in os.listdir(os.path.join(data_root, 'S{}'.format(subject_index), folder_names, "imageSequence", camera)):
                                self.samples.append((subject_index, folder_names, camera, int(frame_index.split('.')[0].split("_")[1])))

    def __getitem__(self, idx):
        subject_index, folder_names, camera, frame_index = self.samples[idx]
        
        img = Image.open(os.path.join(self.data_root, 'S{}'.format(subject_index),
                                      folder_names, "imageSequence", camera, f'img_{frame_index:06d}.jpg'))
        
        img_size = img.size

        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255

        img_array = F.interpolate(img_array[None].float(), size=(512, 512), mode='bilinear', align_corners=False)[0]

        annot_file = h5py.File(os.path.join(self.data_root, 'S{}'.format(subject_index), folder_names, "annot.h5"), "r")
        correct_cam = np.array(annot_file['camera'])==int(camera)
        correct_frame = np.array(annot_file['frame'])==frame_index
        assert (correct_cam*correct_frame).sum() == 1
        annot_frame = np.nonzero(correct_cam*correct_frame)[0][0]
        pose = annot_file['pose/2d'][annot_frame]
        pose /= img_size
        
        # transpose last 2 dimensions
        pose = pose[:, [1, 0]]

        
        
        # import matplotlib.pyplot as plt
        # plt.imshow(img_array.permute(1, 2, 0)) 
        # plt.scatter(pose[:, 1]*512, pose[:, 0]*512)
        # plt.savefig("temp.png")
        # pass

        return {'img': img_array, 'kpts': torch.tensor(pose), 'visibility': torch.ones(pose.shape[0])}

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