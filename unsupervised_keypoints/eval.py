# load the dataset
import os
import torch
import torch.nn.functional as F
from unsupervised_keypoints import ptp_utils
from datasets.celeba import CelebA
from datasets import cub
from datasets import cub_parts
from datasets import taichi
from datasets import human36m
from datasets import unaligned_human36m
from datasets import deepfashion
from unsupervised_keypoints.invertable_transform import (
    RandomAffineWithInverse,
)

# now import weights and biases
import wandb


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

def find_k_max_pixels(map, num=3):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]
    
    output shape [num, batch_size, 2]
    """
    
    batch_size, h, w = map.shape
    
    points = []
    
    for i in range(num):
    
        point = find_max_pixel(map)
        points.append(point)
        
        map = mask_radius(map, point, 0.05*h)

    return torch.stack(points)

def mask_radius(map, max_coords, radius):
    """
    Masks all values within a given radius of the max_coords in the map.
    
    Args:
    map (Tensor): The attention map with shape [batch_size, h, w].
    max_coords (Tensor): The coordinates of the point to mask around, shape [batch_size, 2].
    radius (float): The radius within which to mask the values.
    
    Returns:
    Tensor: The masked map.
    """
    batch_size, h, w = map.shape

    # Create a meshgrid to compute the distance for each pixel
    x_coords = torch.arange(w).view(1, -1).repeat(h, 1).to(map.device)
    y_coords = torch.arange(h).view(-1, 1).repeat(1, w).to(map.device)
    x_coords = x_coords.unsqueeze(0).repeat(batch_size, 1, 1)
    y_coords = y_coords.unsqueeze(0).repeat(batch_size, 1, 1)

    # Calculate squared Euclidean distance from the max_coords
    squared_dist = (x_coords - max_coords[:, 1].unsqueeze(1).unsqueeze(2))**2 + \
                   (y_coords - max_coords[:, 0].unsqueeze(1).unsqueeze(2))**2

    # Mask out pixels within the specified radius
    mask = squared_dist > radius**2
    masked_map = map * mask.float()

    return masked_map

def pixel_from_weighted_avg(heatmaps, distance=5):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]
    """

    # Get the shape of the heatmaps
    batch_size, m, n = heatmaps.shape

    # If distance is provided, zero out elements outside the distance from the max pixel
    if distance != -1:
        # Find max pixel using your existing function or similar logic
        max_pixel_indices = find_max_pixel(heatmaps)
        x_max, y_max = max_pixel_indices[:, 0].long(), max_pixel_indices[:, 1].long()
        
        # Create a meshgrid
        x = torch.arange(0, m).float().view(1, m, 1).to(heatmaps.device).repeat(batch_size, 1, 1)
        y = torch.arange(0, n).float().view(1, 1, n).to(heatmaps.device).repeat(batch_size, 1, 1)
        
        # Calculate the distance to the max_pixel
        distance_to_max = torch.sqrt((x - x_max.view(batch_size, 1, 1)) ** 2 + 
                                     (y - y_max.view(batch_size, 1, 1)) ** 2)
        
        # Zero out elements beyond the distance
        heatmaps[distance_to_max > distance] = 0.0

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

    entropy = entropy.sum(dim=0)

    # get a sorted list of tokens with lowest entropy
    sorted_entropy = torch.argsort(entropy)

    chosen_maps = maps[:, sorted_entropy[:num_points], :, :]

    highest_indices = find_max_pixel(chosen_maps.view(num_images * num_points, h, w))

    highest_indices = highest_indices.reshape(num_images, num_points, 2)

    return highest_indices, sorted_entropy[:num_points]

@torch.no_grad()
def run_image_with_context_augmented(
    ldm,
    image,
    context,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    augmentation_iterations=20,
    noise_level=-1,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    visualize=False,
    controllers=None,
    num_gpus=1,
    save_folder="outputs",
    upscale_size = 512,
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    num_samples = torch.zeros(len(indices), upscale_size, upscale_size).to(device)
    sum_samples = torch.zeros(len(indices), upscale_size, upscale_size).to(device)

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
    )

    if visualize:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(augmentation_iterations + 1, 8)

        visualize_index = 3

    images = []

    for i in range(augmentation_iterations//num_gpus):
        
        augmented_img = (
            invertible_transform(torch.tensor(image)[None].repeat(num_gpus, 1, 1, 1).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
            .numpy()
        )
        
        attn_maps = ptp_utils.run_and_find_attn(
            ldm,
            augmented_img,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upscale_size,
            device=device,
            controllers=controllers,
            indices=indices.cpu(),
        )
        
        attn_maps = torch.stack(attn_maps)
        
        _num_samples = invertible_transform.inverse(torch.ones_like(attn_maps))
        _sum_samples = invertible_transform.inverse(attn_maps)

        num_samples += _num_samples.sum(dim=0)
        sum_samples += _sum_samples.sum(dim=0)

        if visualize:
            inverse_img = invertible_transform.inverse(
                torch.tensor(augmented_img).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
            for j in range(num_gpus):
                index = i*num_gpus+j
                axs[index, 0].imshow(augmented_img[j])
                axs[index, 1].imshow(
                    _num_samples[
                        j, visualize_index, :, :
                    ].cpu()
                )
                axs[index, 2].imshow(attn_maps[j, visualize_index, :, :].cpu())
                axs[index, 3].imshow(
                    _sum_samples[
                        j, visualize_index, :, :
                    ].cpu()
                )
                axs[index, 4].imshow(
                    (
                        attn_maps[j, visualize_index, :, :, None]
                        / attn_maps[j, visualize_index, :, :, None].max()
                    ).cpu()
                    * 0.8
                    + augmented_img[j] * 0.2
                )
                diff = torch.abs(
                    torch.mean(
                        inverse_img[j]
                        .to(device)
                        - torch.tensor(image).to(device),
                        dim=-1,
                    )
                )
                mask = _num_samples[
                    j, 0, None, None
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
                axs[index, 5].imshow((diff)[:, :, None].cpu())
                axs[index, 6].imshow(
                    inverse_img[j]
                    .cpu()
                )
                axs[index, 7].imshow(torch.tensor(image).cpu())

                images.append(
                    inverse_img[j]
                )

    # visualize sum_samples/num_samples
    attention_maps = sum_samples / num_samples

    # replace all nans with 0s
    attention_maps[attention_maps != attention_maps] = 0

    if visualize:
        # import ipdb; ipdb.set_trace()
        re_overlayed_image = torch.sum(torch.stack(images), dim=0).to(device)
        re_overlayed_image /= num_samples[0, :, :, None]
        re_overlayed_image[re_overlayed_image != re_overlayed_image] = 0

        axs[-1, 0].imshow(image)
        axs[-1, 1].imshow(sum_samples[visualize_index].cpu())
        axs[-1, 2].imshow(attention_maps[visualize_index].cpu())
        axs[-1, 3].imshow(re_overlayed_image.cpu())
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
        plt.savefig(os.path.join(save_folder, "augmentation.png"))

    return attention_maps




def swap_points(points):
    """
    Swap keypoints for the human3.6m dataset
    points: B x N x D
    """
    correspondences = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (17, 25), (18, 26), (19, 27), (20, 28), (21, 28), (22, 30), (23, 31)]
    permutation = list(range((points.shape[1])))
    for a, b in correspondences:
        permutation[a] = b
        permutation[b] = a
    new_points = points[:, permutation, :]
    return new_points


@torch.no_grad()
def evaluate(
    ldm,
    context,
    indices,
    regressor,
    args,
    controllers,
    num_gpus,
    from_where=["down_cross", "mid_cross", "up_cross"],
):
    if args.dataset_name == "celeba_aligned":
        dataset = CelebA(split="test", dataset_loc=args.dataset_loc)
    elif args.dataset_name == "celeba_wild":
        dataset = CelebA(split="test", dataset_loc=args.dataset_loc, align = False)
    elif args.dataset_name == "cub_aligned":
        dataset = cub.TestSet(data_root=args.dataset_loc, image_size=512)
    elif args.dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=args.dataset_loc, split="test", single_class=1)
    elif args.dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=args.dataset_loc, split="test", single_class=2)
    elif args.dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=args.dataset_loc, split="test", single_class=3)
    elif args.dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=args.dataset_loc, split="test")
    elif args.dataset_name == "taichi":
        dataset = taichi.TestSet(data_root=args.dataset_loc, image_size=512)
    elif args.dataset_name == "human3.6m":
        dataset = human36m.TestSet(data_root=args.dataset_loc, validation=args.validation)
    elif args.dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TestSet(data_root=args.dataset_loc, image_size=512)
    elif args.dataset_name == "deepfashion":
        dataset = deepfashion.TestSet(data_root=args.dataset_loc, image_size=512)
    else:
        raise NotImplementedError

    distances = []

    max_value = 0

    all_values = []
    
    
    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    dataloader_iter = iter(dataloader)

    for i in range(len(dataset)):

        batch = next(dataloader_iter)

        img = batch["img"][0]

        attention_maps = run_image_with_context_augmented(
            ldm,
            img,
            context,
            indices.cpu(),
            device=args.device,
            from_where=from_where,
            layers=args.layers,
            noise_level=args.noise_level,
            augmentation_iterations=args.augmentation_iterations,
            augment_degrees=args.augment_degrees,
            augment_scale=args.augment_scale,
            augment_translate=args.augment_translate,
            controllers=controllers,
            num_gpus=num_gpus,
            save_folder=args.save_folder,
            visualize=(i==0 and args.visualize),
        )
        
        if args.max_loc_strategy == "argmax":
            highest_indices = find_max_pixel(attention_maps) / 512.0
        else:
            highest_indices = pixel_from_weighted_avg(attention_maps) / 512.0

        # estimated_kpts = regressor(highest_indices.view(-1))
        estimated_kpts = ((highest_indices.view(1, -1)-0.5) @ regressor)+0.5

        estimated_kpts = estimated_kpts.view(-1, 2)

        gt_kpts = batch["kpts"][0].cuda()
        
        if args.evaluation_method == "mean_average_error" or args.evaluation_method == "pck":
            estimated_kpts *= 256
            gt_kpts *= 256

        # get l2 distance between estimated and gt kpts
        l2 = (estimated_kpts - gt_kpts).norm(dim=-1)
        
        if args.evaluation_method == "inter_eye_distance":

            eye_dist = torch.sqrt(torch.sum((gt_kpts[0] - gt_kpts[1]) ** 2, dim=-1))

            l2 = l2 / eye_dist
            
            l2_mean = torch.mean(l2)
            
        if args.evaluation_method == "visible" or args.evaluation_method == "mean_average_error":
            visible = batch['visibility'][0].to(args.device) if 'visibility' in batch else torch.ones_like(l2)
            
            l2_mean = (l2*visible).sum()
            
        if args.evaluation_method == "visible":
            l2_mean /= visible.sum()
            
        if args.evaluation_method == "pck":
            l2_mean = (l2 < 6).float().mean()
            
        if args.evaluation_method == "orientation_invariant":
            l2_mean = l2.mean()
            swapped_kpts = swap_points(estimated_kpts[None])[0]
            
            swapped_l2_mean = (swapped_kpts - gt_kpts).norm(dim=-1).mean()
            
            if swapped_l2_mean < l2_mean:
                l2_mean = swapped_l2_mean
                
            l2_mean *= 128


        

        all_values.append(l2_mean.item())

        if l2_mean > max_value:
            print(f"new max value: {l2_mean}, {i} \n")
            print(i)
            max_value = l2_mean

        distances.append(l2_mean.cpu())
        # eye_dists.append(eye_dist.cpu())

        print(
            f"{(i/len(dataset)):06f}: {i} mean distance: {torch.mean(torch.stack(distances))}, per keypoint: {torch.mean(torch.stack(distances), dim=0)}",
            end="\r",
        )

        if i % 100 == 0:
            print()
        # Extract the 10 worst distances (and their indices) from the priority queue

    if args.wandb:
        wandb.log({"mean_distance": torch.mean(torch.stack(distances))})
    print()

    # save argsorted all_values in torch
    torch.save(torch.tensor(all_values), os.path.join(args.save_folder, "all_errors.pt"))
