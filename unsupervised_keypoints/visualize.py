# load the dataset
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from unsupervised_keypoints import ptp_utils
import torch.nn.functional as F
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints import custom_images
from unsupervised_keypoints import cub
from unsupervised_keypoints import cub_parts
from unsupervised_keypoints import taichi
from unsupervised_keypoints import human36m
from unsupervised_keypoints import unaligned_human36m
from unsupervised_keypoints import deepfashion
from unsupervised_keypoints.eval import run_image_with_context_augmented
from unsupervised_keypoints.eval import pixel_from_weighted_avg, find_max_pixel, mask_radius, find_k_max_pixels

from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse

import matplotlib.pyplot as plt


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


def save_grid(maps, imgs, name, img_size=(512, 512), dpi=50, quality=85):
    """
    There are 10 maps of shape [32, 32]
    There are 10 imgs of shape [3, 512, 512]
    Saves as a single image with matplotlib with 2 rows and 10 columns
    Updated to have smaller borders between images and the edge.
    DPI is reduced to decrease file size.
    JPEG quality can be adjusted to trade off quality for file size.
    """

    # Calculate figure size to maintain aspect ratio
    fig_width = img_size[1] * 10  # total width for 10 images side by side
    fig_height = img_size[0] * 2  # total height for 2 images on top of each other
    fig_size = (fig_width / 100, fig_height / 100)  # scale down to a manageable figure size

    fig, axs = plt.subplots(2, 10, figsize=fig_size, gridspec_kw={'wspace':0.05, 'hspace':0.05})

    for i in range(10):
        axs[0, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        axs[1, i].imshow(imgs[i].numpy().transpose(1, 2, 0))
        normalized_map = maps[i] - torch.min(maps[i])
        normalized_map = normalized_map / torch.max(normalized_map)
        axs[1, i].imshow(normalized_map, alpha=0.7)

    # Remove axis and adjust subplot parameters
    for ax in axs.flatten():
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save as JPEG with reduced DPI and specified quality
    plt.savefig(name, format='jpg', bbox_inches='tight', pad_inches=0, dpi=dpi, pil_kwargs={'quality': quality})

    plt.close()


def plot_point_single(img, points, name):
    """
    Displays corresponding points on the image with white outline around plotted numbers.
    The numbers themselves retain their original color.
    points shape is [num_people, num_points, 2]
    """
    num_people, num_points, _ = points.shape

    # Get the default color cycle from Matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.numpy().transpose(1, 2, 0))

    for i in range(num_people):
        for j in range(num_points):
            # Choose color based on j, cycling through the default color cycle
            color = colors[j % len(colors)]
            x, y = points[i, j, 1] * 512, points[i, j, 0] * 512
            # Plot the original color on top
            ax.scatter(x, y, color=color, marker=f"${j}$", s=300)

    ax.axis("off")  # Remove axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove border

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def plot_point_correspondences(imgs, points, name, height = 11, width = 9):
    """
    Displays corresponding points per image
    len(imgs) = num_images
    points shape is [num_images, num_points, 2]
    """

    num_images, num_points, _ = points.shape

    fig, axs = plt.subplots(height, width, figsize=(2 * width, 2 * height))
    axs = axs.ravel()  # Flatten the 2D array of axes to easily iterate over it

    for i in range(height*width):
        axs[i].imshow(imgs[i].numpy().transpose(1, 2, 0))

        for j in range(num_points):
            # plot the points each as a different type of marker
            axs[i].scatter(
                points[i, j, 1] * 512.0, points[i, j, 0] * 512.0, marker=f"${j}$"
            )

    # remove axis and handle any unused subplots
    for i, ax in enumerate(axs):
        if i >= num_images:
            ax.axis("off")  # Hide unused subplots
        else:
            ax.axis("off")  # Remove axis from used subplots

    # Adjust subplot parameters to reduce space between images and border space
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # increase the resolution of the plot
    plt.savefig(name, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

@torch.no_grad()
def visualize_attn_maps(
    ldm,
    context,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    lr=5e-3,
    noise_level=-1,
    num_tokens=1000,
    num_points=30,
    num_images=100,
    regressor=None,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augmentation_iterations=20,
    dataset_loc="~",
    save_folder="outputs",
    visualize=False,
    dataset_name = "celeba_aligned",
    controllers=None,
    num_gpus=1,
    max_loc_strategy="argmax",
    height = 11,
    width = 9,
    validation = False,
):
    if dataset_name == "celeba_aligned":
        dataset = CelebA(split="test", dataset_loc=dataset_loc)
    elif dataset_name == "celeba_wild":
        dataset = CelebA(split="test", dataset_loc=dataset_loc, align = False)
    elif dataset_name == "cub_aligned":
        dataset = cub.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=1)
    elif dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=2)
    elif dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test", single_class=3)
    elif dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="test")
    elif dataset_name == "taichi":
        dataset = taichi.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "human3.6m":
        dataset = human36m.TestSet(data_root=dataset_loc, validation=validation)
    elif dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "deepfashion":
        dataset = deepfashion.TestSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "custom":
        dataset = custom_images.CustomDataset(data_root=dataset_loc, image_size=512)
    else:
        raise NotImplementedError

    imgs = []
    maps = []
    gt_kpts = []
    
    # random permute the dataset
    randperm = torch.randperm(len(dataset))
    
    for i in tqdm(range(height * width)):
        batch = dataset[randperm[i%len(dataset)].item()]

        img = batch["img"]

        _gt_kpts = batch["kpts"] 
        gt_kpts.append(_gt_kpts)
        imgs.append(img.cpu())

        map = run_image_with_context_augmented(
            ldm,
            img,
            context,
            indices.cpu(),
            device=device,
            from_where=from_where,
            layers=layers,
            noise_level=noise_level,
            augment_degrees=augment_degrees,
            augment_scale=augment_scale,
            augment_translate=augment_translate,
            augmentation_iterations=augmentation_iterations,
            visualize=(i==0),
            controllers=controllers,
            num_gpus=num_gpus,
            save_folder=save_folder,
            human36m=dataset_name == "human3.6m",
        )

        maps.append(map.cpu())
    maps = torch.stack(maps)
    gt_kpts = torch.stack(gt_kpts)

    if max_loc_strategy == "argmax":
        points = find_max_pixel(maps.view(height * width * num_points, 512, 512)) / 512.0
    else:
        points = pixel_from_weighted_avg(maps.view(height * width * num_points, 512, 512)) / 512.0
    points = points.reshape(height * width, num_points, 2)

    plot_point_correspondences(
        imgs, points.cpu(), os.path.join(save_folder, "unsupervised_keypoints.pdf"), height, width
    )

    for i in range(num_points):
        save_grid(
            maps[:, i].cpu(), imgs, os.path.join(save_folder, f"keypoint_{i:03d}.png")
        )

    if regressor is not None:
        est_points = ((points.view(num_images, -1)-0.5) @ regressor)+0.5

        plot_point_correspondences(
            imgs,
            est_points.view(num_images, -1, 2).cpu(),
            os.path.join(save_folder, "estimated_keypoints.pdf"),
            height,
            width,
        )

        plot_point_correspondences(
            imgs, gt_kpts, os.path.join(save_folder, "gt_keypoints.pdf"), height, width
        )