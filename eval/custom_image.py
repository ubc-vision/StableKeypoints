import os
import cv2
import math
import torch
import itertools
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        # torch.tensor([[0.4], [0.9]])*512.0
        source_img = self.load_image("example_images/gnochi_mirror_cropped.png")
        target_img = self.load_image("example_images/cat1.jpeg")
        src_kps = torch.tensor([[0.4, 0.9]])
        n_points = torch.tensor([1])
        
        # # load source image
        # source_img = self.load_image("example_images/bird_7_flip.jpg")
        # target_img = self.load_image("example_images/airplane_2.jpg")
        # source_img = self.load_image("example_images/airplane_2.jpg")
        # target_img = self.load_image("example_images/bird_4.jpg")
        
        # # for lamb_1
        # src_kps = torch.tensor([[0.188725, 0.348974], [0.107843, 0.442815], [0.316176, 0.237537], [0.174020, 0.260997], [0.879902, 0.348974], [0.328431, 0.686217], [0.872549, 0.712610], [0.595588, 0.281525]])
        # n_points = torch.tensor([8])
        # # for airplane_2
        # src_kps = torch.tensor([[0.147059, 0.459459], [0.420588, 0.432432], [0.691176, 0.105105], [0.867647, 0.444444], [0.570588, 0.303303], [0.917647, 0.297297]])
        # n_points = torch.tensor([6])
        # # for chair_1
        # src_kps = torch.tensor([[0.226667, 0.795824], [0.458667, 0.935035], [0.738667, 0.744780], [0.152000, 0.482599], [0.408000, 0.584687], [0.656000, 0.468677], [0.581333, 0.150812]])
        # n_points = torch.tensor([7])
        # for bird_2
        # src_kps = torch.tensor([[0.288000, 0.584000], [0.424000, 0.501333], [0.516000, 0.320000], [0.656000, 0.112000], [0.372000, 0.664000], [0.390000, 0.861333], [0.624000, 0.562667]])
        # # for bird_6
        # src_kps = torch.tensor([[0.278000, 0.327273], [0.462000, 0.368831], [0.658000, 0.114286], [0.742000, 0.462338], [0.682000, 0.532468], [0.176000, 0.807792], [0.650000, 0.415584]])
        # n_points = torch.tensor([7])
        
        # # for bird_7
        # src_kps = torch.tensor([[0.426000, 0.162534], [0.608000, 0.493113], [0.390000, 0.567493], [0.492000, 0.710744], [0.584000, 0.209366], [0.514000, 0.611570]]) 
        
        # src_kps = torch.tensor([[0.422000, 0.191667]])
        # n_points = torch.tensor([1])
        
        src_kps = src_kps.permute(1, 0)*512.0


        return {'pckthres': torch.tensor([1e10]), 'og_src_img': source_img, 'og_trg_img': target_img, 'src_kps': src_kps, 'trg_kps': src_kps, 'n_pts': n_points, 'idx': torch.tensor([0])}

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

