import os
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from unsupervised_keypoints.custom_transform import CustomTransform
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F


# class CelebA(Dataset):
#     """
#     This class is used to create a custom dataset for training and testing the model.
#     """

#     def __init__(self, split="train", augment=False):
#         self.base_folder = "/ubc/cs/home/i/iamerich/scratch/img_align_celeba/"

#         landmark_loc = os.path.join(self.base_folder, "list_landmarks_align_celeba.csv")
#         self.landmarks = pd.read_csv(landmark_loc)

#         partitions_loc = os.path.join(self.base_folder, "list_eval_partition.csv")
#         self.partitions = pd.read_csv(partitions_loc)

#         split_map = {"train": 0, "val": 1, "test": 2}

#         self.len = self.partitions["partition"].value_counts()[split_map[split]]

#         # make the start index the sum of the lengths of the previous splits
#         self.start_index = (
#             self.partitions["partition"].value_counts()[: split_map[split]].sum()
#         )

#         self.num_kps = 5

#         self.augment = augment

#         # if split == "train":
#         #     self.indices = np.arange(0, 162770)
#         # elif split == "test":

#         # # Define a transform pipeline
#         # self.transform = CustomTransform(
#         # degrees=30,  # Random rotation between -30 and 30 degrees
#         # scale=(1.0, 1.0),
#         # translate=(0.0, 0.0),
#         # )
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomAffine(
#                     degrees=30,  # Random rotation between -30 and 30 degrees
#                     scale=(1.0, 1.1),  # Random scaling between 1.0 and 1.2
#                     translate=(0.1, 0.1),  # Random translation by 10% of the image size
#                 ),
#             ]
#         )

#     # def find_indices_for_file_names_random_order(self, file_names):
#     #     shuffled_file_names = (
#     #         file_names.copy()
#     #     )  # Create a copy to avoid modifying the original list
#     #     random.shuffle(shuffled_file_names)  # Shuffle the list in-place

#     #     for file_name in shuffled_file_names:
#     #         yield self.partitions[self.partitions["image_id"] == file_name].index[0]

#     def __len__(self):
#         return self.len

#     def __getitem__(self, index):
#         index = index + self.start_index

#         img = self.load_image(index)

#         kpts = self.load_keypoints(index)

#         if self.augment:
#             img = self.transform(img)

#         return {"img": img, "kpts": kpts}

#     def load_image(self, index):
#         image = Image.open(self.return_img_path(index)).convert("RGB")

#         image = image.resize((512, 512), Image.BILINEAR)

#         image = np.array(image)

#         image = np.transpose(image, (2, 0, 1))

#         image = torch.tensor(image) / 255.0

#         return image

#     def load_keypoints(self, index):
#         width, height = Image.open(self.return_img_path(index)).size

#         landmark = self.landmarks.iloc[index]

#         keypoints = torch.tensor(
#             [
#                 [landmark.lefteye_x, landmark.lefteye_y],
#                 [landmark.righteye_x, landmark.righteye_y],
#                 [landmark.nose_x, landmark.nose_y],
#                 [landmark.leftmouth_x, landmark.leftmouth_y],
#                 [landmark.rightmouth_x, landmark.rightmouth_y],
#             ]
#         )

#         # normalize by image size
#         keypoints = keypoints / torch.tensor([width, height])

#         # swap the x and y
#         keypoints = keypoints[:, [1, 0]]

#         return keypoints

#     def return_img_path(self, index):
#         img_name = self.landmarks.iloc[index].image_id

#         return os.path.join(self.base_folder, "img_align_celeba", img_name)


class CelebA(Dataset):
    """
    This class is used to create a custom dataset for training and testing the model.
    """

    def __init__(self, split="train", align=True):
        self.base_folder = "/ubc/cs/home/i/iamerich/scratch/datasets/celeba/"

        self.mafl_loc = (
            "/ubc/cs/home/i/iamerich/scratch/datasets/celeba/TCDCN-face-alignment/MAFL/"
        )

        if align:
            landmark_loc = os.path.join(
                self.base_folder, "Anno", "list_landmarks_align_celeba.txt"
            )
        else:
            landmark_loc = os.path.join(
                self.base_folder, "Anno", "list_landmarks_celeba.txt"
            )

        # load the .txt file
        self.landmarks = open(landmark_loc, "r")
        self.landmarks = self.landmarks.readlines()

        self.num_kps = 5

        self.align = align

        if split == "test":
            self.file_names = open(os.path.join(self.mafl_loc, "testing.txt"), "r")
        elif split == "train":
            self.file_names = open(os.path.join(self.mafl_loc, "training.txt"), "r")
        self.file_names = self.file_names.readlines()

        # # Define a transform pipeline
        # self.transform = CustomTransform(
        # degrees=30,  # Random rotation between -30 and 30 degrees
        # scale=(1.0, 1.0),
        # translate=(0.0, 0.0),
        # )
        self.transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=30,  # Random rotation between -30 and 30 degrees
                    scale=(1.0, 1.1),  # Random scaling between 1.0 and 1.2
                    translate=(0.1, 0.1),  # Random translation by 10% of the image size
                ),
            ]
        )

    def __len__(self):
        return len(self.file_names)

    def find_local_index(self, global_index):
        local_file_name = self.file_names[global_index]
        # remove everything after the "."
        local_file_name = local_file_name.split(".")[0]

        # convert to int
        local_file_name = int(local_file_name)

        # convert to 0 indexing
        local_file_name = local_file_name - 1

        return local_file_name

    def __getitem__(self, index):
        local_index = self.find_local_index(index)

        img = self.load_image(local_index)

        kpts = self.load_keypoints(local_index)

        return {"img": img, "kpts": kpts}

    def load_image(self, index):
        image = Image.open(self.return_img_path(index)).convert("RGB")

        image = image.resize((512, 512), Image.BILINEAR)

        image = np.array(image)

        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image) / 255.0

        return image

    def load_keypoints(self, index):
        width, height = Image.open(self.return_img_path(index)).size

        # Get the line corresponding to the index
        line = self.landmarks[index + 2]  # +2 to skip the header lines

        # Split the line by spaces and ignore the image name
        parts = line.split()[1:]

        # Convert to numbers
        keypoints = [float(p) for p in parts]

        # Reshape keypoints into [5, 2] and convert to torch tensor
        keypoints = torch.tensor(keypoints).reshape(5, 2)

        # normalize by image size
        keypoints = keypoints / torch.tensor([width, height])

        # swap the x and y
        keypoints = keypoints[:, [1, 0]]

        return keypoints

    def return_img_path(self, index):
        # img_name = self.landmarks.iloc[index].image_id

        img_name = f"{index+1:06d}" + (".png" if self.align else ".jpg")

        if self.align:
            return os.path.join(
                self.base_folder, "Img", "img_align_celeba_png", img_name
            )
        else:
            return os.path.join(self.base_folder, "Img", "img_celeba", img_name)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse

    ds = CelebA(split="test", align=False)

    transform = RandomAffineWithInverse(
        degrees=30, scale=(1.0, 1.1), translate=(0.1, 0.1)
    )

    img = ds[999]["img"]

    transformed_img = transform(img)

    initial_image = transform.inverse(transformed_img)

    # plot all of img, transformed_img, and initial_image in the same figure
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img.permute(1, 2, 0).cpu().detach().numpy())
    axs[1].imshow(transformed_img.permute(1, 2, 0).cpu().detach().numpy())
    axs[2].imshow(initial_image.permute(1, 2, 0).cpu().detach().numpy())
    plt.savefig(f"outputs/image.png")
    plt.close()

    # print(len(ds))

    # mini_batch = ds[999]

    # image = mini_batch["img"]
    # keypoints = mini_batch["kpts"]

    # plt.imshow(image.permute(1, 2, 0).cpu().detach().numpy())
    # # plot the keypoints on the image
    # plt.scatter(
    #     keypoints[:, 1] * 512.0, keypoints[:, 0] * 512.0, marker="x", color="red"
    # )
    # plt.savefig(f"outputs/image.png")
    # plt.close()
