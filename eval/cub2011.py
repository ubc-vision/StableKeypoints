# import os
# import pandas as pd
# from torchvision.datasets.folder import default_loader
# from torchvision.datasets.utils import download_url
# from torch.utils.data import Dataset


# class Cub2011(Dataset):
#     base_folder = 'CUB_200_2011/images'
#     url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
#     filename = 'CUB_200_2011.tgz'
#     tgz_md5 = '97eceeb196236b17998738112f37df78'

#     def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.loader = default_loader
#         self.train = train

#         if download:
#             self._download()

#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')

#     def _load_metadata(self):
#         images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
#                              names=['img_id', 'filepath'])
#         image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
#                                          sep=' ', names=['img_id', 'target'])
#         train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
#                                        sep=' ', names=['img_id', 'is_training_img'])

#         data = images.merge(image_class_labels, on='img_id')
#         self.data = data.merge(train_test_split, on='img_id')

#         if self.train:
#             self.data = self.data[self.data.is_training_img == 1]
#         else:
#             self.data = self.data[self.data.is_training_img == 0]

#     def _check_integrity(self):
#         try:
#             self._load_metadata()
#         except Exception:
#             return False

#         for index, row in self.data.iterrows():
#             filepath = os.path.join(self.root, self.base_folder, row.filepath)
#             if not os.path.isfile(filepath):
#                 print(filepath)
#                 return False
#         return True

#     def _download(self):
#         import tarfile

#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return

#         download_url(self.url, self.root, self.filename, self.tgz_md5)

#         with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
#             tar.extractall(path=self.root)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data.iloc[idx]
#         path = os.path.join(self.root, self.base_folder, sample.filepath)
#         target = sample.target - 1  # Targets start at 1 by default, so shift to 0
#         img = self.loader(path)

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, target
    
# if __name__ == "__main__":
#     # load the cub dataset 
#     cub = Cub2011(root = "/scratch/iamerich/Download/", train=False)
    
#     # get the next item
#     img, target = cub.__getitem__(0)
#     pass


import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, root_dir, train=True, num_classes=3):
        self.root_dir = root_dir
        self.train = train
        self.num_classes = num_classes

        # Load image list
        with open(os.path.join(root_dir, "images.txt"), "r") as f:
            self.images = [line.strip().split() for line in f.readlines()]

        # Load train/test split
        with open(os.path.join(root_dir, "train_test_split.txt"), "r") as f:
            self.train_test_split = [line.strip().split() for line in f.readlines()]

        # Load part locations
        with open(os.path.join(root_dir, "parts/part_locs.txt"), "r") as f:
            self.part_locs = {}
            for line in f.readlines():
                img_id, part_id, x, y, visible = line.strip().split()
                if img_id not in self.part_locs:
                    self.part_locs[img_id] = []
                self.part_locs[img_id].append((int(part_id), float(x), float(y), int(visible)))

        # Load image class labels
        with open(os.path.join(root_dir, "image_class_labels.txt"), "r") as f:
            self.image_class_labels = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

        # Filter images based on train/test split and class labels
        self.filtered_images = []
        for img_id, img_name in self.images:
            is_training_image = int(self.train_test_split[int(img_id) - 1][1])
            class_id = self.image_class_labels[img_id]
            if ((self.train and is_training_image) or (not self.train and not is_training_image)) and (class_id <= self.num_classes):
                self.filtered_images.append((img_id, img_name))
                
    def __len__(self):
        return len(self.filtered_images)

    def __getitem__(self, idx):
        img_id, img_name = self.filtered_images[idx]
        img_path = os.path.join(self.root_dir, "images", img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # Convert image to PyTorch tensor

        # pad the outside of image with 0s to make it 512x512
        image = torch.nn.functional.pad(image, (0, 512 - image.shape[2], 0, 512 - image.shape[1]))

        # Load keypoints and visibility
        keypoints = torch.tensor(self.part_locs[img_id], dtype=torch.float)
        visibility = keypoints[:, -1].bool()

        return image, keypoints[:, 1:3], visibility

    # rest of the code remains the same

# Creating DataLoader for training and testing sets
root_dir = "/scratch/iamerich/Download/CUB_200_2011"

test_dataset = CustomDataset(root_dir, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# get the next batch from train_dataloader
img, keypoints, visibility = next(iter(test_dataloader))
pass