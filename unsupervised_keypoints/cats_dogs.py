import os
import cv2
import math
import torch
import itertools
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


class Cats_Dogs(Dataset):
    """
    This class is used to create a custom dataset for training and testing the model.
    """
    def __init__(self, train = True, dogs = True, *args, **kwargs):
        self.dogs = dogs
        self.cuties = glob(f"example_images/cats_and_dogs_filtered/{'train' if train else 'validation'}/{'dogs' if dogs else 'cats'}/*.jpg")
        
    def __len__(self):
        return len(self.cuties)

    def __getitem__(self, idx):
        img = self.load_image(self.cuties[idx])

        return {'img': img, "label": 1 if self.dogs else 0}

    def load_image(self, img_name):
        image = Image.open(img_name).convert('RGB')

        image = image.resize((512, 512), Image.BILINEAR)
    
        image = np.array(image)
    
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image)/255.0

        return image


if __name__ == "__main__":
    # Creating DataLoader for training and testing sets
    root_dir = "/scratch/iamerich/Datasets_CATs"

    test_dataset = CustomDataset(root_dir, train=False)
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
    