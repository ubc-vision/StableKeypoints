import os
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

# from unsupervised_keypoints.custom_transform import CustomTransform
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F


class CelebA(Dataset):
    """
    This class is used to create a custom dataset for training and testing the model.
    """

    def __init__(
        self,
        split="train",
        align=True,
        mafl_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/TCDCN-face-alignment/MAFL/",
        celeba_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    ):
        self.celeba_loc = celeba_loc
        self.mafl_loc = mafl_loc

        if align:
            landmark_loc = os.path.join(
                self.celeba_loc, "Anno", "list_landmarks_align_celeba.txt"
            )
        else:
            landmark_loc = os.path.join(
                self.celeba_loc, "Anno", "list_landmarks_celeba.txt"
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
                self.celeba_loc, "Img", "img_align_celeba_png", img_name
            )
        else:
            return os.path.join(self.celeba_loc, "Img", "img_celeba", img_name)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse

    ds = CelebA(align=True, split="test")

    transform = RandomAffineWithInverse(
        # degrees=100,
        scale=(0.5, 0.5),
        # translate=(0.0, 1.0),
        translate=(2.0, 2.0),
    )

    img = ds[121]["img"]
    kpts = ds[121]["kpts"]

    # transformed_img, transformed_kpts = transform(img, kpts)
    transformed_img = transform(img)
    transformed_kpts = transform.transform_keypoints(kpts, 512)

    initial_image = transform.inverse(transformed_img)
    initial_keypoints_prime = transform.inverse_transform_keypoints(
        transformed_kpts, 512
    )

    # plot all of img, transformed_img, and initial_image in the same figure
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img.permute(1, 2, 0).cpu().detach().numpy())
    # plot the keypoints on the image (shape is [5, 2])
    axs[0].scatter(kpts[:, 1] * 512.0, kpts[:, 0] * 512.0, marker="x", color="red")
    axs[1].imshow(transformed_img.permute(1, 2, 0).cpu().detach().numpy())
    axs[1].scatter(
        transformed_kpts[:, 1] * 512.0,
        transformed_kpts[:, 0] * 512.0,
        marker="x",
        color="red",
    )
    axs[2].imshow(initial_image.permute(1, 2, 0).cpu().detach().numpy())
    axs[2].scatter(
        initial_keypoints_prime[:, 1] * 512.0,
        initial_keypoints_prime[:, 0] * 512.0,
        marker="x",
        color="red",
    )
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
