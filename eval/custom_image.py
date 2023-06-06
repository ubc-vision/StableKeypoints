import os
import cv2
import math
import torch
import itertools
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    This class is used to create a custom dataset for training and testing the model.
    """
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        source_img = self.load_image("example_images/source_cat.png")
        target_img = self.load_image("example_images/target_cat.jpeg")
        src_kps = torch.tensor([[0.4, 0.9]])
        trg_kps = torch.tensor([[0.54, 0.92]])
        n_points = torch.tensor([1])
        
        src_kps = src_kps.permute(1, 0)*512.0
        trg_kps = trg_kps.permute(1, 0)*512.0


        return {'pckthres': torch.tensor([512.0]), 'og_src_img': source_img, 'og_trg_img': target_img, 'src_kps': src_kps, 'trg_kps': trg_kps, 'n_pts': n_points, 'idx': torch.tensor([0])}

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
    