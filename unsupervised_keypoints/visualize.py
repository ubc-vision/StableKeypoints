# load the dataset
import torch
import numpy as np
from tqdm import tqdm
from unsupervised_keypoints import ptp_utils
import torch.nn.functional as F
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints.eval import run_image_with_tokens_augmented
from unsupervised_keypoints.eval import pixel_from_weighted_avg, find_max_pixel

from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse

import matplotlib.pyplot as plt

# from scipy.ndimage import zoom

# # now import weights and biases
# import wandb


# from unsupervised_keypoints.optimize_token import (
#     init_random_noise,
#     image2latent,
#     AttentionStore,
# )

# from unsupervised_keypoints.optimize import collect_maps
# from unsupervised_keypoints.eval import get_attn_map


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

        axs[2, i].imshow(normalized_map, alpha=0.7)

    # remove axis
    for ax in axs.flatten():
        ax.axis("off")
    plt.savefig(f"outputs/{name}_grid.png")
    plt.close()


def plot_point_correspondences(imgs, points, name):
    """
    Displays corresponding points per image
    len(imgs) = num_images
    points shape is [num_images, num_points, 2]
    """

    num_images, num_points, _ = points.shape

    # points is shape [num_images, num_selected_tokens, 2]
    num_images = len(imgs)

    fig, axs = plt.subplots(1, 10, figsize=(2 * num_images, 2))
    for i in range(num_images):
        axs[i].imshow(imgs[i].numpy().transpose(1, 2, 0))

        for j in range(num_points):
            # plot the points each as a different type of marker
            axs[i].scatter(
                points[i, j, 1] * 512.0, points[i, j, 0] * 512.0, marker=f"${j}$"
            )

    # remove axis
    for ax in axs.flatten():
        ax.axis("off")
    # increase the resolution of the plot
    plt.savefig(f"outputs/{name}_grid.png", dpi=300)
    # plt.savefig(f"outputs/{name}_grid.png")
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
    num_images=10,
    regressor=None,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augment_shear=(0.0, 0.0),
    augmentation_iterations=20,
):
    dataset = CelebA(split="test")

    imgs = []
    maps = []
    gt_kpts = []
    # for i in tqdm(range(num_images)):
    for i in [527, 115, 121, 815, 753, 528, 487, 210, 263, 778]:
        batch = dataset[i]

        img = batch["img"]

        _gt_kpts = batch["kpts"]
        gt_kpts.append(_gt_kpts)
        imgs.append(img.cpu())

        map = run_image_with_tokens_augmented(
            ldm,
            img,
            context,
            indices,
            device=device,
            from_where=from_where,
            layers=layers,
            noise_level=noise_level,
            augment_degrees=augment_degrees,
            augment_scale=augment_scale,
            augment_translate=augment_translate,
            augment_shear=augment_shear,
            augmentation_iterations=augmentation_iterations,
            visualize=True,
        )

        maps.append(map)
    maps = torch.stack(maps)
    gt_kpts = torch.stack(gt_kpts)

    points = find_max_pixel(maps.view(num_images * num_points, 512, 512)) / 512.0
    points = points.reshape(num_images, num_points, 2)

    plot_point_correspondences(imgs, points.cpu(), "unsupervised_keypoints")

    for i in range(num_points):
        save_grid(maps[:, i].cpu(), imgs, f"keypoint_{i:03d}")

    if regressor is not None:
        est_points = points.view(num_images, -1) @ regressor

        plot_point_correspondences(
            imgs,
            est_points.view(num_images, 5, 2).cpu(),
            "estimated_keypoints",
        )

        plot_point_correspondences(imgs, gt_kpts, "gt_keypoints")

        pass
