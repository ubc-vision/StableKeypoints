

# load the dataset
import torch
import numpy as np
from tqdm import tqdm
from unsupervised_keypoints import ptp_utils
import torch.nn.functional as F
from unsupervised_keypoints.cats_dogs import Cats_Dogs
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints.eval import find_corresponding_points

import matplotlib.pyplot as plt

from scipy.ndimage import zoom

# now import weights and biases
import wandb


from unsupervised_keypoints.optimize_token import init_random_noise, image2latent, AttentionStore

from unsupervised_keypoints.optimize import collect_maps
from unsupervised_keypoints.eval import get_attn_map

def save_img(map, img, name):
    # save with matplotlib
    # map is shape [32, 32]
    import matplotlib.pyplot as plt
    plt.imshow(map.cpu().detach().numpy())
    plt.title(f"max: {torch.max(map).cpu().detach().numpy()}")
    plt.savefig(f"outputs/{name}_map.png")
    plt.close()
    # save original image current with shape [3, 512, 512]
    plt.imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
    plt.savefig(f"outputs/{name}_img.png")
    plt.close()
    
def save_grid(maps, imgs, name):
    """
    There are 10 maps of shape [32, 32] 
    There are 10 imgs of shape [3, 512, 512]
    Saves as a single image with matplotlib with 2 rows and 10 columns
    """
    
    fig, axs = plt.subplots(3, 10, figsize=(15, 4))
    for i in range(10):
        axs[0, i].imshow(maps[i].numpy())
        axs[0, i].set_title(f"max: {torch.max(maps[i]).numpy():.2f}")
        axs[1, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        axs[2, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        normalized_map = maps[i] - torch.min(maps[i])
        normalized_map = normalized_map / torch.max(normalized_map)
        
        map_resized = zoom(normalized_map, (512/32, 512/32))  # resize the heatmap
        
        axs[2, i].imshow(map_resized, alpha=0.7)
    
    # remove axis
    for ax in axs.flatten():
        ax.axis('off')
    plt.savefig(f"outputs/{name}_grid.png")
    plt.close()
    
    
def plot_point_correspondences(imgs, maps, name, num_points=10):
    """
    Displays corresponding points per image
    len(imgs) = num_images
    maps shape is [num_images, num_tokens, 32, 32]
    """
    
    # points is shape [num_images, num_selected_tokens, 2]
    points, indices = find_corresponding_points(maps, num_points=num_points)
    
    num_images = len(imgs)
    
    
    fig, axs = plt.subplots(1, 10, figsize=(2*num_images, 2))
    for i in range(num_images):
        axs[i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        
        for j in range(num_points):
            # plot the points each as a different type of marker
            axs[i].scatter(points[i, j, 1]/32*512, points[i, j, 0]/32*512, marker=f"${j}$")
    
    # remove axis
    for ax in axs.flatten():
        ax.axis('off')
    plt.savefig(f"outputs/{name}_grid.png")
    plt.close()

def visualize(ldm, context, device="cuda", num_steps=2000, from_where = ["down_cross", "mid_cross", "up_cross"], upsample_res = 32, layers = [0, 1, 2, 3, 4, 5], lr=5e-3, noise_level = -1, num_tokens = 77):
    
    # dogs = Cats_Dogs(dogs=True)
    # cats = Cats_Dogs(dogs=False, train=False)
    dataset = CelebA()
    
    if context is None:
        context = init_random_noise(device)
        
    context.requires_grad = True
    
    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)
    
    imgs = []
    maps = []
    for i in tqdm(range(10)):
        batch = dataset[i]
        img = batch['img']
        imgs.append(img.cpu())
    
        map = get_attn_map(ldm, img, context, device = device, from_where = from_where, upsample_res = upsample_res, layers = layers, noise_level = noise_level, num_tokens = num_tokens)
        maps.append(map.detach().cpu())
    maps = torch.stack(maps)
    
    # # make the borders of the image (the final 2 dimensions) zero
    maps[:, :, :, :3] = 0
    maps[:, :, :, -3:] = 0
    maps[:, :, :3, :] = 0
    maps[:, :, -3:, :] = 0
    
    
    # import torch.distributions as dist
        
    # # Normalize the activation maps to represent probability distributions
    # attention_maps_softmax = torch.softmax(maps.view(10*num_tokens, -1), dim=-1)

    # # Compute the entropy of each token
    # entropy = dist.Categorical(probs=attention_maps_softmax).entropy()
    
    
    # entropy = entropy.reshape(10, num_tokens)

    # # get a sorted list of tokens with lowest entropy
    # sorted_entropy = torch.argsort(entropy, dim=-1)
    
    # ranks = torch.argsort(sorted_entropy, dim=-1)
    
    points, indices = find_corresponding_points(maps, num_points=10)
    
    for i in tqdm(indices):
        
        save_grid(maps[:, i], imgs, f"keypoint_{i:03d}")
            
    plot_point_correspondences(imgs, maps, "correspondences")