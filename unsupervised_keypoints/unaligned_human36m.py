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
from torchvision.transforms import functional as TF


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

def crop_and_upsample(img_array, pose, margin=100, jitter = 100, target_size=(512, 512)):
    """
    Crop the image based on the normalized keypoints from pose, apply a margin, introduce random jitter,
    and then upsample it bilinearly. Also, adjust the keypoints according to the cropped and jittered image.

    Parameters:
    img_array (torch.Tensor): The image tensor of shape (C, H, W).
    pose (torch.Tensor): The normalized pose keypoints tensor of shape (N, 2) with values from 0 to 1.
    margin (int): The number of pixels to add as margin to the bounding box.
    jitter (int): The maximum number of pixels to use as translation jitter when cropping.
    target_size (tuple): The target size to upsample the image to.

    Returns:
    torch.Tensor: The cropped and upsampled image tensor.
    torch.Tensor: The new keypoints adjusted to the cropped and jittered image.
    """
    # Ensure pose is a torch tensor and switch x and y
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)

    # Denormalize the keypoints to the image size
    pose[:, 0] *= img_array.shape[1]  # Height
    pose[:, 1] *= img_array.shape[2]  # Width

    # Calculate min and max of the keypoints for x and y
    x_min, y_min = torch.min(pose, dim=0)[0]
    x_max, y_max = torch.max(pose, dim=0)[0]

    # Determine the bounding box size (making it square based on the max dimension)
    width = x_max - x_min
    height = y_max - y_min
    side_length = max(width, height)

    # Calculate margin, ensuring it doesn't exceed the image dimensions
    margin_x = min(margin, img_array.shape[2] - side_length)
    margin_y = min(margin, img_array.shape[1] - side_length)

    # Introduce random jitter within the specified range
    jitter_x = torch.randint(-jitter, jitter, (1,)).item()
    jitter_y = torch.randint(-jitter, jitter, (1,)).item()

    # Adjust bounding box with margin and jitter
    x_min = max(0, x_min - (side_length - width) / 2 - margin_x + jitter_x)
    y_min = max(0, y_min - (side_length - height) / 2 - margin_y + jitter_y)

    x_max = min(img_array.shape[2], x_min + side_length + 2 * margin_x)
    y_max = min(img_array.shape[1], y_min + side_length + 2 * margin_y)

    # Crop the image
    cropped_img = TF.crop(img_array, int(y_min), int(x_min), int(y_max - y_min), int(x_max - x_min))

    # Update keypoints based on the crop and jitter
    # The jitter needs to be subtracted from the keypoints because the image is moving in the opposite direction
    new_pose = pose - torch.tensor([[x_min, y_min]])

    # Normalize the keypoints by the cropped size
    new_pose[:, 1] /= (y_max - y_min)
    new_pose[:, 0] /= (x_max - x_min)

    # Upsample the image
    upsampled_img = F.interpolate(cropped_img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    
    # Adjust keypoints to the upsampled size
    new_pose[:, 0] *= target_size[0]
    new_pose[:, 1] *= target_size[1]

    return upsampled_img.squeeze(0), new_pose/512


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
        
        img_size = img.size

        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255

        annot_file = h5py.File(os.path.join(self.data_root, 'S{}'.format(subject_index), folder_names, "annot.h5"), "r")
        correct_cam = np.array(annot_file['camera'])==int(camera)
        correct_frame = np.array(annot_file['frame'])==frame_index
        assert (correct_cam*correct_frame).sum() == 1
        annot_frame = np.nonzero(correct_cam*correct_frame)[0][0]
        _pose = annot_file['pose/2d'][annot_frame]
        _pose /= img_size
        
        img_array, pose = crop_and_upsample(img_array, torch.tensor(_pose))
        
        # transpose last 2 dimensions
        pose = pose[:, [1, 0]]
        
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

        annot_file = h5py.File(os.path.join(self.data_root, 'S{}'.format(subject_index), folder_names, "annot.h5"), "r")
        correct_cam = np.array(annot_file['camera'])==int(camera)
        correct_frame = np.array(annot_file['frame'])==frame_index
        assert (correct_cam*correct_frame).sum() == 1
        annot_frame = np.nonzero(correct_cam*correct_frame)[0][0]
        _pose = annot_file['pose/2d'][annot_frame]
        _pose /= img_size
        
        img_array, pose = crop_and_upsample(img_array, torch.tensor(_pose))
        
        # transpose last 2 dimensions
        pose = pose[:, [1, 0]]

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

        annot_file = h5py.File(os.path.join(self.data_root, 'S{}'.format(subject_index), folder_names, "annot.h5"), "r")
        correct_cam = np.array(annot_file['camera'])==int(camera)
        correct_frame = np.array(annot_file['frame'])==frame_index
        assert (correct_cam*correct_frame).sum() == 1
        annot_frame = np.nonzero(correct_cam*correct_frame)[0][0]
        _pose = annot_file['pose/2d'][annot_frame]
        _pose /= img_size
        
        img_array, pose = crop_and_upsample(img_array, torch.tensor(_pose))
        
        # transpose last 2 dimensions
        pose = pose[:, [1, 0]]

        return {'img': img_array, 'kpts': torch.tensor(pose), 'visibility': torch.ones(pose.shape[0])}

    def __len__(self):
        return len(self.samples)