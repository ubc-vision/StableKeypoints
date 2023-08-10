import os
import cv2
import math
import torch
import itertools
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class CelebA(Dataset):
    """
    This class is used to create a custom dataset for training and testing the model.
    """
    def __init__(self, *args, **kwargs):
        self.images = glob(f"/ubc/cs/home/i/iamerich/scratch/celeb_a_hq/celeba-512/*.jpg")
        
        # Define a transform pipeline
        self.transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=30,  # Random rotation between -30 and 30 degrees
                    scale=(1.0, 1.1),  # Random scaling between 1.0 and 1.2
                    translate=(0.1, 0.1)  # Random translation by 10% of the image size
                ),
            ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.load_image(self.images[idx])

        return {'img': img}

    def load_image(self, img_name):
        image = Image.open(img_name).convert('RGB')
        
        image = self.transform(image)

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
    