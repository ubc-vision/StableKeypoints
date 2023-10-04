# load the dataset
import os
import torch
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
import torch.nn.functional as F
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints.cub import TestSet
from unsupervised_keypoints.invertable_transform import (
    RandomAffineWithInverse,
    return_theta,
)

# now import weights and biases
import wandb

# from unsupervised_keypoints.optimize_token import init_random_noise

from unsupervised_keypoints import optimize


def save_img(map, img, point, name):
    # save with matplotlib
    # map is shape [32, 32]
    import matplotlib.pyplot as plt

    plt.imshow(map.cpu().detach().numpy())
    plt.title(f"max: {torch.max(map).cpu().detach().numpy()}")
    # plot point on image
    plt.scatter(point[1].cpu() * 512, point[0].cpu() * 512, c="r")
    plt.savefig(f"outputs/{name}_map.png")
    plt.close()
    # save original image current with shape [3, 512, 512]
    plt.imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
    plt.scatter(point[1].cpu() * 512, point[0].cpu() * 512, c="r")
    plt.savefig(f"outputs/{name}_img.png")
    plt.close()


def find_max_pixel(map):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]
    
    output shape [batch_size, 2]
    """

    batch_size, h, w = map.shape

    map_reshaped = map.view(batch_size, -1)

    max_indices = torch.argmax(map_reshaped, dim=-1)

    max_indices = max_indices.view(batch_size, 1)

    max_indices = torch.cat([max_indices // w, max_indices % w], dim=-1)

    # offset by a half a pixel to get the center of the pixel
    max_indices = max_indices + 0.5

    return max_indices


def pixel_from_weighted_avg(heatmaps):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]
    """

    # Get the shape of the heatmaps
    batch_size, m, n = heatmaps.shape

    # Compute the total value of the heatmaps
    total_value = torch.sum(heatmaps, dim=[1, 2], keepdim=True)

    # Normalize the heatmaps
    normalized_heatmaps = heatmaps / (
        total_value + 1e-6
    )  # Adding a small constant to avoid division by zero

    # Create meshgrid to represent the coordinates
    x = torch.arange(0, m).float().view(1, m, 1).to(heatmaps.device)
    y = torch.arange(0, n).float().view(1, 1, n).to(heatmaps.device)

    # Compute the weighted sum for x and y
    x_sum = torch.sum(x * normalized_heatmaps, dim=[1, 2])
    y_sum = torch.sum(y * normalized_heatmaps, dim=[1, 2])

    return torch.stack([x_sum, y_sum], dim=-1) + 0.5


def find_corresponding_points(maps, num_points=10):
    """
    Finds the corresponding points between the maps.
    Selects a set of maps with the common lowest entropy
    Argmax of each of these is the corresponding point

    map shape [num_images, num_tokens, h, w]

    returns maximum pixel location and indices of the maps with the lowest entropy
    """

    num_images, num_tokens, h, w = maps.shape

    import torch.distributions as dist

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(
        maps.view(num_images * num_tokens, h * w), dim=-1
    )

    # Compute the entropy of each token
    entropy = dist.Categorical(probs=attention_maps_softmax).entropy()

    entropy = entropy.reshape(num_images, num_tokens)

    # _, top_embedding_indices = torch.topk(entropy, num_tokens, largest=False)

    entropy = entropy.sum(dim=0)

    # get a sorted list of tokens with lowest entropy
    sorted_entropy = torch.argsort(entropy)

    chosen_maps = maps[:, sorted_entropy[:num_points], :, :]

    highest_indices = find_max_pixel(chosen_maps.view(num_images * num_points, h, w))

    highest_indices = highest_indices.reshape(num_images, num_points, 2)

    return highest_indices, sorted_entropy[:num_points]


# def crop_image(image, crop_percent=90):
#     """pixel is an integer between 0 and image.shape[1] or image.shape[2]"""

#     assert 0 < crop_percent <= 100, "crop_percent should be between 0 and 100"

#     height, width, channels = image.shape
#     crop_height = int(height * crop_percent / 100)
#     crop_width = int(width * crop_percent / 100)

#     x_start_max = width - crop_width
#     y_start_max = height - crop_height

#     # Choose a random top-left corner within the allowed bounds
#     x_start = torch.randint(0, int(x_start_max) + 1, (1,)).item()
#     y_start = torch.randint(0, int(y_start_max) + 1, (1,)).item()

#     # Crop the image
#     cropped_image = image[
#         y_start : y_start + crop_height, x_start : x_start + crop_width
#     ]

#     # bilinearly upsample to 512x512
#     cropped_image = torch.nn.functional.interpolate(
#         torch.tensor(cropped_image[None]).permute(0, 3, 1, 2),
#         size=(512, 512),
#         mode="bilinear",
#         align_corners=False,
#     )[0]

#     return (
#         cropped_image.permute(1, 2, 0).numpy(),
#         y_start,
#         crop_height,
#         x_start,
#         crop_width,
#     )


# @torch.no_grad()
# def progressively_zoom_into_image(
#     ldm,
#     image,
#     context,
#     indices,
#     device="cuda",
#     from_where=["down_cross", "mid_cross", "up_cross"],
#     layers=[0, 1, 2, 3, 4, 5],
#     num_zooms=2,
#     noise_level=-1,
#     visualize=False,
#     rotation_degrees=15,
# ):
#     """
#     First forward passes the image with no augmentation and find the argmax for each keypoint
#     Then 'zoom' in on each keypoint by cropping the image around the keypoint
#     """

#     num_samples = torch.zeros(len(indices), 512, 512).to(device)
#     sum_samples = torch.zeros(len(indices), 512, 512).to(device)

#     points = []

#     # if image is a torch.tensor, convert to numpy
#     if type(image) == torch.Tensor:
#         image = image.permute(1, 2, 0).detach().cpu().numpy()

#     if parent_keypoints is None:
#         initial_maps = ptp_utils.run_and_find_attn(
#             ldm,
#             image,
#             context,
#             layers=layers,
#             noise_level=noise_level,
#             from_where=from_where,
#             upsample_res=512,
#             indices=indices,
#         )

#         highest_indices, confidences = find_max_pixel(
#             initial_maps, return_confidences=True
#         )
#         highest_indices = highest_indices / 512.0

#     else:
#         highest_indices = parent_keypoints.clone()

#     if visualize:
#         import matplotlib.pyplot as plt

#         fig, axs = plt.subplots(4, 11)
#         axs[0, 0].imshow(image)
#         axs[1, 0].imshow(image)
#         for i in range(highest_indices.shape[0]):
#             # make the point the number f"{i}"

#             axs[0, 0].scatter(
#                 highest_indices[i, 1].cpu() * 512,
#                 highest_indices[i, 0].cpu() * 512,
#                 marker=f"${i}$",
#             )

#     transform = RandomAffineWithInverse()

#     for keypoint in range(highest_indices.shape[0]):
#         # randomly choose rotation between -rotation_degrees and rotation_degrees
#         random_rot = torch.rand(1) * 2 * rotation_degrees - rotation_degrees

#         theta = return_theta(0.5, highest_indices[keypoint], random_rot)

#         augmented_img = (
#             transform(torch.tensor(image).permute(2, 0, 1), theta)
#             .permute(1, 2, 0)
#             .numpy()
#         )

#         maps = ptp_utils.run_and_find_attn(
#             ldm,
#             augmented_img,
#             context,
#             layers=layers,
#             noise_level=noise_level,
#             from_where=from_where,
#             upsample_res=512,
#             indices=indices,
#         )

#         # transform all keypoints to this view to see which are in view
#         # untransform the maps to the original view and add the maps which are within view

#         transformed_highest_indices = transform.transform_keypoints(highest_indices)
#         within_view = (
#             (transformed_highest_indices > 0.1) * (transformed_highest_indices < 0.9)
#         ).sum(dim=1) == 2

#         sum_samples[within_view] += transform.inverse(maps)[within_view]
#         num_samples[within_view] += transform.inverse(torch.ones_like(maps))[
#             within_view
#         ]

#         highest_indices_iteration, confidences = find_max_pixel(
#             maps, return_confidences=True
#         )
#         highest_indices_iteration = highest_indices_iteration / 512.0

#         inverted_kpts = transform.inverse_transform_keypoints(highest_indices_iteration)

#         points.append(inverted_kpts[keypoint])

#         if visualize:
#             axs[0, keypoint + 1].scatter(
#                 256,
#                 256,
#                 marker=f"${keypoint}$",
#             )

#             axs[0, keypoint + 1].scatter(
#                 highest_indices_iteration[keypoint, 1].cpu() * 512,
#                 highest_indices_iteration[keypoint, 0].cpu() * 512,
#                 marker=f"${keypoint}$",
#             )

#             axs[0, keypoint + 1].imshow(augmented_img)

#             # upscale maps to 512x512
#             maps_upscaled = torch.nn.functional.interpolate(
#                 maps[keypoint, None, None],
#                 size=(512, 512),
#                 mode="bilinear",
#                 align_corners=False,
#             )[0, 0]
#             axs[1, keypoint + 1].imshow(maps_upscaled.cpu(), alpha=0.5)
#             axs[1, keypoint + 1].imshow(augmented_img, alpha=0.5)

#             axs[1, keypoint + 1].scatter(
#                 highest_indices_iteration[keypoint, 1].cpu() * 512,
#                 highest_indices_iteration[keypoint, 0].cpu() * 512,
#                 marker=f"${keypoint}$",
#             )

#             axs[1, 0].scatter(
#                 inverted_kpts[keypoint, 1].cpu() * 512,
#                 inverted_kpts[keypoint, 0].cpu() * 512,
#                 marker=f"${keypoint}$",
#             )

#     points = torch.stack(points)

#     attention_maps = sum_samples / num_samples
#     # replace all nans with 0s
#     attention_maps[attention_maps != attention_maps] = 0

#     if visualize:
#         for i in range(points.shape[0]):
#             axs[1, 0].scatter(
#                 points[i, 1].cpu() * 512,
#                 points[i, 0].cpu() * 512,
#                 marker=f"${i}$",
#             )

#             axs[2, i + 1].imshow(attention_maps[i].cpu())
#             axs[3, i + 1].imshow((num_samples[i] / num_samples[i].max()).cpu())
#         # # increase resolution of pyplot to 512x2048
#         fig.set_size_inches(4096 / 100, 4096 / 400)
#         plt.savefig("initial.png")

#         pass

#     return points


@torch.no_grad()
def run_image_with_tokens_augmented(
    ldm,
    image,
    tokens,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    augmentation_iterations=20,
    noise_level=-1,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augment_shear=(0.0, 0.0),
    visualize=False,
    controller=None,
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    num_samples = torch.zeros(len(indices), 512, 512).to(device)
    sum_samples = torch.zeros(len(indices), 512, 512).to(device)

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
        shear=augment_shear,
    )

    if visualize:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(augmentation_iterations + 1, 8)

        visualize_index = 3

    images = []

    for i in range(augmentation_iterations):
        augmented_img = (
            invertible_transform(torch.tensor(image).permute(2, 0, 1))
            .permute(1, 2, 0)
            .numpy()
        )

        # if i != 0:
        #     augmented_img = (
        #         invertible_transform(torch.tensor(image).permute(2, 0, 1))
        #         .permute(1, 2, 0)
        #         .numpy()
        #     )
        # else:
        #     augmented_img = image

        latents = ptp_utils.image2latent(ldm, augmented_img, device)

        latents = ldm.scheduler.add_noise(
            latents, torch.rand_like(latents), ldm.scheduler.timesteps[noise_level]
        )

        latents = ptp_utils.diffusion_step(
            ldm,
            controller,
            latents,
            tokens,
            ldm.scheduler.timesteps[noise_level],
            cfg=False,
        )

        _attention_maps, _ = optimize.collect_maps(
            controller,
            from_where=from_where,
            upsample_res=512,
            layers=layers,
            indices=indices,
        )

        num_samples += invertible_transform.inverse(torch.ones_like(num_samples))
        sum_samples += invertible_transform.inverse(_attention_maps)

        if visualize:
            axs[i, 0].imshow(augmented_img)
            axs[i, 1].imshow(
                invertible_transform.inverse(torch.ones_like(num_samples))[
                    visualize_index, :, :
                ].cpu()
            )
            axs[i, 2].imshow(_attention_maps[visualize_index, :, :].cpu())
            axs[i, 3].imshow(
                invertible_transform.inverse(_attention_maps)[
                    visualize_index, :, :
                ].cpu()
            )
            axs[i, 4].imshow(
                (
                    _attention_maps[visualize_index, :, :, None]
                    / _attention_maps[visualize_index, :, :, None].max()
                ).cpu()
                * 0.8
                + augmented_img * 0.2
            )
            diff = torch.abs(
                torch.mean(
                    invertible_transform.inverse(
                        torch.tensor(augmented_img).permute(2, 0, 1)
                    )
                    .permute(1, 2, 0)
                    .to(device)
                    - torch.tensor(image).to(device),
                    dim=-1,
                )
            )
            mask = invertible_transform.inverse(torch.ones_like(num_samples))[
                0, None, None
            ].to(device)
            kernel = (
                torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
                .reshape(1, 1, 3, 3)
                .to(device)
            )
            mask = F.conv2d(mask, kernel, padding=1)
            mask = (mask == 9).float()

            diff *= mask[0, 0]
            diff = diff / diff.max()
            axs[i, 5].imshow((diff)[:, :, None].cpu())
            axs[i, 6].imshow(
                invertible_transform.inverse(
                    torch.tensor(augmented_img).permute(2, 0, 1)
                )
                .permute(1, 2, 0)
                .cpu()
            )
            axs[i, 7].imshow(torch.tensor(image).cpu())

            images.append(
                invertible_transform.inverse(
                    torch.tensor(augmented_img).permute(2, 0, 1)
                )
            )

    # visualize sum_samples/num_samples
    attention_maps = sum_samples / num_samples

    # replace all nans with 0s
    attention_maps[attention_maps != attention_maps] = 0

    if visualize:
        re_overlayed_image = torch.sum(torch.stack(images), dim=0).to(device)
        re_overlayed_image /= num_samples[0, None]
        re_overlayed_image[re_overlayed_image != re_overlayed_image] = 0

        axs[-1, 0].imshow(image)
        axs[-1, 1].imshow(sum_samples[visualize_index].cpu())
        axs[-1, 2].imshow(attention_maps[visualize_index].cpu())
        axs[-1, 3].imshow(re_overlayed_image.cpu().permute(1, 2, 0))
        axs[-1, 4].imshow(
            (
                attention_maps[visualize_index, :, :, None]
                / attention_maps[visualize_index, :, :, None].max()
            ).cpu()
            * 0.8
            + image * 0.2
        )

        # set the resolution of the plot to 512x512
        fig.set_size_inches(4096 / 100, 4096 / 100)
        plt.savefig("augmentation.png")

    return attention_maps


@torch.no_grad()
def evaluate(
    ldm,
    context,
    indices,
    regressor,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    noise_level=-1,
    num_tokens=1000,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augment_shear=(0.0, 0.0),
    augmentation_iterations=20,
    mafl_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/TCDCN-face-alignment/MAFL/",
    celeba_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    cub_loc="/ubc/cs/home/i/iamerich/scratch/datasets/cub/cub",
    save_folder="outputs",
    wandb_log=False,
    visualize=False,
    dataset_name = "celeba_aligned",
    evaluation_method="inter_eye_distance",
    controller=None,
):
    if dataset_name == "celeba_aligned":
        dataset = CelebA(split="test", mafl_loc=mafl_loc, celeba_loc=celeba_loc)
    elif dataset_name == "celeba_wild":
        dataset = CelebA(split="test", mafl_loc=mafl_loc, celeba_loc=celeba_loc, align = False)
    elif dataset_name == "cub":
        dataset = TestSet(data_root=cub_loc, image_size=512)
    else:
        raise NotImplementedError

    distances = []

    # eye_dists = []

    worst_l2 = PriorityQueue()

    max_value = 0

    all_values = []

    for i in range(len(dataset)):
        batch = dataset[i]

        img = batch["img"]

        attention_maps = run_image_with_tokens_augmented(
            ldm,
            img,
            context,
            indices,
            device=device,
            from_where=from_where,
            layers=layers,
            noise_level=noise_level,
            augmentation_iterations=augmentation_iterations,
            augment_degrees=augment_degrees,
            augment_scale=augment_scale,
            augment_translate=augment_translate,
            augment_shear=augment_shear,
            controller=controller,
            # visualize=True,
        )
        highest_indices = find_max_pixel(attention_maps)
        highest_indices = highest_indices / 512.0

        # estimated_kpts = regressor(highest_indices.view(-1))
        estimated_kpts = ((highest_indices.view(1, -1)-0.5) @ regressor)+0.5

        estimated_kpts = estimated_kpts.view(-1, 2)

        gt_kpts = batch["kpts"].cuda()


        # get l2 distance between estimated and gt kpts
        l2 = (estimated_kpts - gt_kpts).norm(dim=-1)
        
        if evaluation_method == "inter_eye_distance":

            eye_dist = torch.sqrt(torch.sum((gt_kpts[0] - gt_kpts[1]) ** 2, dim=-1))

            l2 = l2 / eye_dist
            
            l2_mean = torch.mean(l2)
            
        if evaluation_method == "visible":
            visible = batch['visibility'].to(device)
            
            l2_mean = (l2*visible).sum()/visible.sum()


        

        all_values.append(l2_mean.item())

        if l2_mean > max_value:
            print(f"new max value: {l2_mean}, {i} \n")
            print(i)
            max_value = l2_mean

        if worst_l2.qsize() < 10:
            worst_l2.put((l2_mean.item(), i))
        else:
            smallest_worst, smallest_worst_index = worst_l2.get()
            if l2_mean.item() > smallest_worst:
                worst_l2.put((l2_mean.item(), i))
            else:
                worst_l2.put((smallest_worst, smallest_worst_index))

        distances.append(l2_mean.cpu())
        # eye_dists.append(eye_dist.cpu())

        print(
            f"{(i/len(dataset)):06f}: {i} mean distance: {torch.mean(torch.stack(distances))}, per keypoint: {torch.mean(torch.stack(distances), dim=0)}",
            end="\r",
        )

        if i % 100 == 0:
            print()
        # Extract the 10 worst distances (and their indices) from the priority queue

    if wandb_log:
        wandb.log({"mean_distance": torch.mean(torch.stack(distances))})
    print()

    worst_10 = []
    while not worst_l2.empty():
        distance, index = worst_l2.get()
        worst_10.append((index, distance))

    # Now worst_10 contains the indices and l2 distances of the 10 worst cases
    print("10 worst L2 distances and their indices:")
    for index, distance in reversed(worst_10):
        print(f"Index: {index}, L2 Distance: {distance}")

    print()

    # save argsorted all_values in torch
    torch.save(torch.tensor(all_values), os.path.join(save_folder, "argsort_test.pt"))
