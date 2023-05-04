import os
import cv2
import math
import torch
import itertools
import numpy as np
from PIL import Image, ImageOps
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
        img1, scale_factor_1, pad_left_1, pad_top_1, bool_img_src = self.load_image(img1_name)
        img2, scale_factor_2, pad_left_2, pad_top_2, bool_img_trg = self.load_image(img2_name)

        # Load keypoints and visibility
        keypoints1 = torch.tensor(self.part_locs[img1_id], dtype=torch.float)
        keypoints1[:, 1] *= scale_factor_1[0]
        keypoints1[:, 2] *= scale_factor_1[1]
        keypoints1[:, 1] += pad_left_1
        keypoints1[:, 2] += pad_top_1
        visibility1 = keypoints1[:, -1].bool()
        keypoints2 = torch.tensor(self.part_locs[img2_id], dtype=torch.float)
        keypoints2[:, 1] *= scale_factor_2[0]
        keypoints2[:, 2] *= scale_factor_2[1]
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
        pck_threshold = self.compute_pck_threshold_per_image(bbox, scale_factor_2[0])

        return {'pckthres': pck_threshold, 'og_src_img': img1/255.0, 'og_trg_img': img2/255.0, 'src_kps': reordered_keypoints1.permute(1, 0), 'trg_kps': reordered_keypoints2.permute(1, 0), 'n_pts': num_overlapping, 'bbox': bbox, 'idx': idx, 'bool_img_src':bool_img_src, 'bool_img_trg':bool_img_trg}

    def load_image(self, img_name):
        img_path = os.path.join(self.datapath, "images", img_name)
        image = Image.open(img_path).convert('RGB')

        width, height = image.size
        max_dim = max(width, height)
        scale_factor = 512 / max_dim
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        pad_left = (512 - new_width) // 2
        pad_right = 512 - new_width - pad_left
        pad_top = (512 - new_height) // 2
        pad_bottom = 512 - new_height - pad_top

        image_np = np.array(image)
        tiled_image = cv2.copyMakeBorder(image_np, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_WRAP)

        image = torch.from_numpy(tiled_image).permute(2, 0, 1).float()

        # Create a boolean image
        bool_image = torch.zeros((512, 512), dtype=torch.bool)
        bool_image[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = 1

        return image, torch.tensor([scale_factor, scale_factor], dtype=torch.float), pad_left, pad_top, bool_image
        
    def compute_pck_threshold_per_image(self, bbox, scale_factor=1.0):
        
        width, height = bbox[2], bbox[3]
        pck_threshold = max(width, height)
        return pck_threshold*scale_factor


if __name__ == "__main__":
    # Creating DataLoader for training and testing sets
    root_dir = "/scratch/iamerich/Datasets_CATs"

    test_dataset = CUBDataset(root_dir, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # # get the next batch from train_dataloader
    batch = next(iter(test_dataloader))
    # # img, keypoints, visibility = next(iter(test_dataloader))


    # visualize the image with matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(batch['og_src_img'][0].permute(1, 2, 0).numpy())
    plt.scatter(batch['src_kps'][0, 0], batch['src_kps'][0, 1], c='r', s=10)
    plt.savefig("img1.png")
    plt.close()

    import matplotlib.pyplot as plt
    plt.imshow(batch['og_trg_img'][0].permute(1, 2, 0).numpy())
    plt.scatter(batch['trg_kps'][0, 0], batch['trg_kps'][0, 1], c='r', s=10)
    plt.savefig("img2.png")
    plt.close()
    
    import matplotlib.pyplot as plt
    plt.imshow(batch['bool_img_trg'].permute(1, 2, 0).numpy())
    plt.scatter(batch['trg_kps'][0, 0], batch['trg_kps'][0, 1], c='r', s=10)
    plt.savefig("img2_bool.png")
    plt.close()
    
    import matplotlib.pyplot as plt
    plt.imshow(batch['bool_img_src'].permute(1, 2, 0).numpy())
    plt.scatter(batch['src_kps'][0, 0], batch['src_kps'][0, 1], c='r', s=10)
    plt.savefig("img1_bool.png")
    plt.close()

    pass

