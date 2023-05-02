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
import math
import torch
import itertools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CUBDataset(Dataset):
    def __init__(self, datapath="/scratch/iamerich/Datasets_CATs", split="test", num_classes=3, item_index=-1, *args, **kwargs):
        self.datapath = f"{datapath}/CUB_200_2011"
        self.train = split!="test"
        self.num_classes = num_classes

        # Load image list
        with open(os.path.join(self.datapath, "images.txt"), "r") as f:
            self.images = [line.strip().split() for line in f.readlines()]

        # Load train/test split
        with open(os.path.join(self.datapath, "train_test_split.txt"), "r") as f:
            self.train_test_split = [line.strip().split() for line in f.readlines()]

        # Load part locations
        with open(os.path.join(self.datapath, "parts/part_locs.txt"), "r") as f:
            self.part_locs = {}
            for line in f.readlines():
                img_id, part_id, x, y, visible = line.strip().split()
                if img_id not in self.part_locs:
                    self.part_locs[img_id] = []
                self.part_locs[img_id].append((int(part_id), float(x), float(y), int(visible)))

        # Load image class labels
        with open(os.path.join(self.datapath, "image_class_labels.txt"), "r") as f:
            self.image_class_labels = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

        # Filter images based on train/test split and class labels
        self.filtered_images = []
        for img_id, img_name in self.images:
            is_training_image = int(self.train_test_split[int(img_id) - 1][1])
            class_id = self.image_class_labels[img_id]
            if ((self.train and is_training_image) or (not self.train and not is_training_image)) and (class_id <= self.num_classes):
                self.filtered_images.append((img_id, img_name))
        
        # Generate all pairs for each class
        pairs = []
        for class_id in range(1, self.num_classes + 1):
            class_images = [img for img in self.filtered_images if self.image_class_labels[img[0]] == class_id]
            class_pairs = list(itertools.combinations(class_images, 2))
            pairs.extend(class_pairs)

        # Use only the specified pair index if provided
        if item_index != -1:
            if 0 <= item_index < len(pairs):
                self.pairs = [pairs[item_index]]
            else:
                raise IndexError("The specified pair index is out of range.")
        else:
            self.pairs = pairs
            
        # Load bounding box data
        with open(os.path.join(self.datapath, "bounding_boxes.txt"), "r") as f:
            self.bounding_boxes = {line.split()[0]: list(map(float, line.strip().split()[1:])) for line in f.readlines()}
                
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (img1_id, img1_name), (img2_id, img2_name) = self.pairs[idx]

        # Load images
        img1, pad_left_1, pad_top_1 = self.load_image(img1_name)
        img2, pad_left_2, pad_top_2 = self.load_image(img2_name)

        # Load keypoints and visibility
        keypoints1 = torch.tensor(self.part_locs[img1_id], dtype=torch.float)
        keypoints1[:, 1] += pad_left_1
        keypoints1[:, 2] += pad_top_1
        visibility1 = keypoints1[:, -1].bool()
        keypoints2 = torch.tensor(self.part_locs[img2_id], dtype=torch.float)
        keypoints2[:, 1] += pad_left_2
        keypoints2[:, 2] += pad_top_2
        visibility2 = keypoints2[:, -1].bool()

        # Find overlapping visible keypoints
        overlapping = visibility1 & visibility2
        num_overlapping = overlapping.sum().item()
        num_total_keypoints = keypoints1.shape[0]

        # Create new keypoints tensors with overlapping keypoints first, followed by -1s
        reordered_keypoints1 = torch.full((num_total_keypoints, 2), -1, dtype=torch.float)
        reordered_keypoints2 = torch.full((num_total_keypoints, 2), -1, dtype=torch.float)

        reordered_keypoints1[:num_overlapping] = keypoints1[overlapping, 1:3]
        reordered_keypoints2[:num_overlapping] = keypoints2[overlapping, 1:3]
        
        # Load bounding box for the image
        bbox = self.bounding_boxes[img2_id]

        # Compute PCK threshold for the image
        pck_threshold = self.compute_pck_threshold_per_image(bbox)

        return {'pckthres': pck_threshold, 'og_src_img': img1/255.0, 'og_trg_img': img2/255.0, 'src_kps': reordered_keypoints1.permute(1, 0), 'trg_kps': reordered_keypoints2.permute(1, 0), 'n_pts': num_overlapping, 'bbox': bbox, 'idx': idx}

    def load_image(self, img_name):
        img_path = os.path.join(self.datapath, "images", img_name)
        image = Image.open(img_path).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        padding_left = (512 - image.shape[2]) // 2
        padding_right = 512 - image.shape[2] - padding_left
        padding_top = (512 - image.shape[1]) // 2
        padding_bottom = 512 - image.shape[1] - padding_top

        image = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top, padding_bottom))
        return image, padding_left, padding_top
    
    def compute_pck_threshold_per_image(self, bbox):
        
        width, height = bbox[2], bbox[3]
        pck_threshold = max(width, height)
        return pck_threshold


if __name__ == "__main__":
    # Creating DataLoader for training and testing sets
    root_dir = "/scratch/iamerich/Datasets_CATs/CUB_200_2011"

    test_dataset = CUB2011(root_dir, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # # get the next batch from train_dataloader
    batch = next(iter(test_dataloader))
    # # img, keypoints, visibility = next(iter(test_dataloader))


    # visualize the image with matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(batch['og_src_img'][0].permute(1, 2, 0).numpy().astype(np.uint8))
    plt.scatter(batch['src_kps'][0, :, 0], batch['src_kps'][0, :, 1], c='r', s=10)
    plt.savefig("img1.png")
    plt.close()

    import matplotlib.pyplot as plt
    plt.imshow(batch['og_trg_img'][0].permute(1, 2, 0).numpy().astype(np.uint8))
    plt.scatter(batch['trg_kps'][0, :, 0], batch['trg_kps'][0, :, 1], c='r', s=10)
    plt.savefig("img2.png")
    plt.close()

    pass

