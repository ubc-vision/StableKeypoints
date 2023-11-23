# load the dataset
import torch
import numpy as np
from tqdm import tqdm
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints import sdxl_monkey_patch
from unsupervised_keypoints import eval
import torch.nn.functional as F
import torch.distributions as dist
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints import custom_images
from unsupervised_keypoints import cub
from unsupervised_keypoints import cub_parts
from unsupervised_keypoints import taichi
from unsupervised_keypoints import human36m
from unsupervised_keypoints import unaligned_human36m
from unsupervised_keypoints import deepfashion
from unsupervised_keypoints import optimize_token
import torch.nn as nn

# now import weights and biases
import wandb

from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse


# from unsupervised_keypoints.optimize_token import init_random_noise


def collect_maps(
    controller,
    from_where=["up_cross"],
    upsample_res=512,
    layers=[0, 1, 2, 3],
    indices=None,
):
    """
    returns the bilinearly upsampled attention map of size upsample_res x upsample_res for the first word in the prompt
    """

    attention_maps = controller.step_store['attn']

    attention_maps_list = []

    layer_overall = -1

    for layer in range(len(attention_maps)):
        layer_overall += 1

        if layer_overall not in layers:
            continue

        data = attention_maps[layer]

        data = data.reshape(
            data.shape[0], int(data.shape[1] ** 0.5), int(data.shape[1] ** 0.5), data.shape[2]
        )
        
        # import ipdb; ipdb.set_trace()

        if indices is not None:
            data = data[:, :, :, indices]

        data = data.permute(0, 3, 1, 2)

        if upsample_res != -1 and data.shape[1] ** 0.5 != upsample_res:
            # bilinearly upsample the image to attn_sizexattn_size
            data = F.interpolate(
                data,
                size=(upsample_res, upsample_res),
                mode="bilinear",
                align_corners=False,
            )

        attention_maps_list.append(data)


    attention_maps_list = torch.stack(attention_maps_list, dim=0).mean(dim=(0, 1))

    controller.reset()

    return attention_maps_list


def create_gaussian_kernel(size: int, sigma: float):
    """
    Create a 2D Gaussian kernel of given size and sigma.

    Args:
        size (int): The size (width and height) of the kernel. Should be odd.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        Tensor: A 2D tensor representing the Gaussian kernel.
    """
    assert size % 2 == 1, "Size must be odd"
    center = size // 2

    x = torch.arange(0, size, dtype=torch.float32)
    y = torch.arange(0, size, dtype=torch.float32)
    x, y = torch.meshgrid(x - center, y - center)

    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel


def gaussian_loss(attn_map, kernel_size=5, sigma=1.0, temperature=1e-4):
    # attn_map is of shape (T, H, W)
    T, H, W = attn_map.shape

    # Softmax over flattened attn_map to get probabilities
    attn_probs = F.softmax(attn_map.view(T, -1) / temperature, dim=1)  # Shape: (T, H*W)

    # stop the gradient for attn_probs
    attn_probs = attn_probs.detach()
    # # divide attn_probs by the max of the first dim
    # attn_probs = attn_probs / attn_probs.max(dim=1, keepdim=True)[0]
    attn_probs = attn_probs.view(T, H, W)  # Reshape back to original shape

    # Create Gaussian kernel
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(attn_map.device)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # Apply Gaussian smoothing
    target = F.conv2d(
        attn_probs.unsqueeze(1), gaussian_kernel, padding=kernel_size // 2
    )
    target = target.view(T, H * W)
    target = target / target.max(dim=1, keepdim=True)[0]
    target = target.view(T, H, W)
    # divide attn_probs by the max of the first dim

    loss = F.mse_loss(attn_map, attn_probs)
    # loss = F.mse_loss(attn_probs, torch.zeros_like(attn_probs))

    return loss


def find_pos_from_index(attn_map):
    T, H, W = attn_map.shape

    index = attn_map.view(T, -1).argmax(dim=1)

    # Convert 1D index to 2D indices
    rows = index // W
    cols = index % W

    # Normalize to [0, 1]
    rows_normalized = rows.float() / (H - 1)
    cols_normalized = cols.float() / (W - 1)

    # Combine into pos
    pos = torch.stack([cols_normalized, rows_normalized], dim=1)

    return pos


def equivariance_loss(embeddings_initial, embeddings_transformed, transform, index):
    # untransform the embeddings_transformed
    embeddings_initial_prime = transform.inverse(embeddings_transformed)[index]

    loss = F.mse_loss(embeddings_initial, embeddings_initial_prime)

    return loss


def olf_sharpening_loss(attn_map, kernel_size=5, sigma=1.0, temperature=1e-1, l1=False):
    # attn_map is of shape (T, H, W)
    T, H, W = attn_map.shape

    # Scale attn_map by temperature
    attn_map_scaled = (
        attn_map / temperature
    )  # Adding channel dimension, shape (T, 1, H, W)

    # Apply spatial softmax over attn_map to get probabilities
    spatial_softmax = torch.nn.Softmax2d()
    attn_probs = spatial_softmax(attn_map_scaled)  # Removing channel dimension

    # Find argmax and create one-hot encoding
    argmax_indices = attn_probs.view(T, -1).argmax(dim=1)
    one_hot = torch.zeros_like(attn_probs.view(T, -1)).scatter_(
        1, argmax_indices.view(-1, 1), 1
    )
    one_hot = one_hot.view(T, H, W)

    # Create Gaussian kernel
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(attn_map.device)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # Apply Gaussian smoothing to the one-hot encoding
    target = F.conv2d(one_hot.unsqueeze(1), gaussian_kernel, padding=kernel_size // 2)
    target = target.view(T, H * W)
    target = target / target.max(dim=1, keepdim=True)[0]
    target = target.view(T, H, W)

    # Compute loss
    if l1:
        loss = F.l1_loss(attn_probs, target)
    else:
        loss = F.mse_loss(attn_probs, target)
    # loss = nn.L1Loss()(attn_probs, target)

    return loss


def sharpening_loss(attn_map, sigma=1.0, temperature=1e1, device="cuda", num_subjects = 1):
    
    pos = eval.find_k_max_pixels(attn_map, num=num_subjects)/attn_map.shape[-1]

    loss = find_gaussian_loss_at_point(
        attn_map,
        pos,
        sigma=sigma,
        temperature=temperature,
        device=device,
        num_subjects=num_subjects,
    )

    return loss


def find_gaussian_loss_at_point(
    attn_map, pos, sigma=1.0, temperature=1e-1, device="cuda", indices=None, num_subjects=1
):
    """
    pos is a location between 0 and 1
    """

    # attn_map is of shape (T, H, W)
    T, H, W = attn_map.shape

    # Create Gaussian circle at the given position
    target = optimize_token.gaussian_circles(
        pos, size=H, sigma=sigma, device=attn_map.device
    )  # Assuming H and W are the same
    target = target.to(attn_map.device)

    # possibly select a subset of indices
    if indices is not None:
        attn_map = attn_map[indices]
        target = target[indices]

    # Compute loss
    loss = F.mse_loss(attn_map, target)

    return loss


def variance_loss(heatmaps):
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
    x_sum = torch.sum(x * normalized_heatmaps, dim=[1, 2], keepdim=True)
    y_sum = torch.sum(y * normalized_heatmaps, dim=[1, 2], keepdim=True)

    # Compute the weighted average for x and y
    x_avg = x_sum
    y_avg = y_sum

    # Compute the variance sum
    variance_sum = torch.sum(
        normalized_heatmaps * (((x - x_avg) ** 2) + ((y - y_avg) ** 2)), dim=[1, 2]
    )

    # Compute the standard deviation
    std_dev = torch.sqrt(variance_sum)

    return torch.mean(std_dev)


def spreading_loss(heatmaps, temperature=1e-1):
    # Scale attn_map by temperature
    heatmaps = heatmaps / temperature

    # spatial_softmax = torch.nn.Softmax2d()

    heatmaps = F.softmax(heatmaps.view(heatmaps.shape[0], -1), dim=1).view(
        heatmaps.shape
    )

    # heatmaps = spatial_softmax(heatmaps)  # Removing channel dimension

    locs = differentiable_argmax(heatmaps)

    # Compute the pairwise distance between each pair of points
    total_dist = 0
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0]):
            total_dist += torch.norm(locs[i] - locs[j])

    # we want to maximize the distance between the points
    return -total_dist / (locs.shape[0] * (locs.shape[0] - 1))


def differentiable_argmax(heatmaps):
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

    locs = torch.stack([x_sum, y_sum], dim=1)

    return locs


def optimize_embedding(
    ldm,
    top_k_strategy="entropy",
    wandb_log=True,
    context=None,
    device="cuda",
    num_steps=2000,
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=256,
    layers=[0, 1, 2, 3, 4, 5],
    lr=5e-3,
    noise_level=-1,
    num_tokens=1000,
    top_k=10,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    sdxl=False,
    dataset_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    sigma=1.0,
    sharpening_loss_weight=100,
    equivariance_attn_loss_weight=100,
    batch_size=4,
    num_gpus=1,
    dataset_name = "celeba_aligned",
    max_len=-1,
    min_dist=0.05,
    furthest_point_num_samples=50,
    controllers=None,
    validation = False,
    num_subjects=1,
):
    
    if dataset_name == "celeba_aligned":
        dataset = CelebA(split="train", dataset_loc=dataset_loc, max_len=max_len)
    elif dataset_name == "celeba_wild":
        dataset = CelebA(split="train", dataset_loc=dataset_loc, align = False, max_len=max_len)
    elif dataset_name == "cub_aligned":
        dataset = cub.TrainSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train", single_class=1)
    elif dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train", single_class=2)
    elif dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train", single_class=3)
    elif dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train")
    elif dataset_name == "taichi":
        dataset = taichi.TrainSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "human3.6m":
        dataset = human36m.TrainSet(data_root=dataset_loc, validation=validation)
    elif dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TrainSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "deepfashion":
        dataset = deepfashion.TrainSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "custom":
        dataset = custom_images.CustomDataset(data_root=dataset_loc, image_size=512)
    else:
        raise NotImplementedError


    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
    )

    # every iteration return image, pixel_loc

    if context is None:
        context = ptp_utils.init_random_noise(device, num_words=num_tokens)

    context.requires_grad = True

    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)

    # time the optimization
    import time

    start = time.time()
    it_start = time.time()

    running_equivariance_attn_loss = 0
    running_sharpening_loss = 0
    running_total_loss = 0
    
    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_gpus, shuffle=True, drop_last=True)

    dataloader_iter = iter(dataloader)
    
    # import ipdb; ipdb.set_trace()  
    
    for iteration in tqdm(range(int(num_steps*(batch_size//num_gpus)))):
        
        try:
            mini_batch = next(dataloader_iter)
        except StopIteration:  # Explicitly catch StopIteration
            dataloader_iter = iter(dataloader)
            mini_batch = next(dataloader_iter)

        image = mini_batch["img"]

        attn_maps = ptp_utils.run_and_find_attn(
            ldm,
            image,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=-1,
            device=device,
            controllers=controllers,
            human36m=dataset_name == "human3.6m",
        )
        
        # import ipdb; ipdb.set_trace()

        transformed_img = invertible_transform(image)

        attention_maps_transformed = ptp_utils.run_and_find_attn(
            ldm,
            transformed_img,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=-1,
            device=device,
            controllers=controllers,
            human36m=dataset_name == "human3.6m",
        )
        
        _sharpening_loss = []
        _loss_equivariance_attn = []
        
        for index, attn_map, attention_map_transformed in zip(torch.arange(num_gpus), attn_maps, attention_maps_transformed):

            if top_k_strategy == "entropy":
                top_embedding_indices = ptp_utils.entropy_sort(
                    attn_map, furthest_point_num_samples,
                )
            elif top_k_strategy == "gaussian":
                top_embedding_indices = ptp_utils.find_top_k_gaussian(
                    attn_map, furthest_point_num_samples, sigma=sigma, num_subjects = num_subjects
                )
            elif top_k_strategy == "consistent":
                top_embedding_indices = torch.arange(furthest_point_num_samples)
            else:
                raise NotImplementedError
            
            top_embedding_indices = ptp_utils.furthest_point_sampling(attention_map_transformed, top_k, top_embedding_indices)

            _sharpening_loss.append(sharpening_loss(attn_map[top_embedding_indices], device=device, sigma=sigma, num_subjects = num_subjects))

            _loss_equivariance_attn.append(equivariance_loss(
                attn_map[top_embedding_indices], attention_map_transformed[top_embedding_indices][None].repeat(num_gpus, 1, 1, 1), invertible_transform, index
            ))
        


        _sharpening_loss = torch.stack([loss.to('cuda:0') for loss in _sharpening_loss]).mean()
        _loss_equivariance_attn = torch.stack([loss.to('cuda:0') for loss in _loss_equivariance_attn]).mean()
        

        # use the old loss for the first 1000 iterations
        # new loss is unstable for early iterations
        loss = (
            + _loss_equivariance_attn * equivariance_attn_loss_weight
            # + _spreading_loss * spreading_loss_weight
            + _sharpening_loss * sharpening_loss_weight
        )

        running_equivariance_attn_loss += _loss_equivariance_attn / (batch_size//num_gpus) * equivariance_attn_loss_weight
        running_sharpening_loss += _sharpening_loss / (batch_size//num_gpus) * sharpening_loss_weight
        # running_spreading_loss += _spreading_loss / (batch_size//num_gpus)
        running_total_loss += loss / (batch_size//num_gpus)

        loss = loss / (batch_size//num_gpus)

        loss.backward()
        if (iteration + 1) % (batch_size//num_gpus) == 0:
            optimizer.step()
            optimizer.zero_grad()

            if wandb_log:
                wandb.log(
                    {
                        "loss": running_total_loss.item(),
                        "running_equivariance_attn_loss": running_equivariance_attn_loss.item(),
                        "running_sharpening_loss": running_sharpening_loss.item(),
                        "iteration time": time.time() - it_start,
                        # "running_spreading_loss": running_spreading_loss.item(),
                    }
                )
            else:
                print(
                    f"loss: {loss.item()}, \
                    _loss_equivariance_attn: {running_equivariance_attn_loss.item()} \
                    sharpening_loss: {running_sharpening_loss.item()},  \
                    running_total_loss: {running_total_loss.item()}, \
                    iteration time: {time.time() - it_start}"
                )
            running_equivariance_attn_loss = 0
            running_sharpening_loss = 0
            running_total_loss = 0
            
            it_start = time.time()

    print(f"optimization took {time.time() - start} seconds")

    return context.detach()

