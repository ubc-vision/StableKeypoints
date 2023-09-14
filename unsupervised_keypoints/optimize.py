# load the dataset
import torch
import numpy as np
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints import sdxl_monkey_patch
from unsupervised_keypoints import eval
import torch.nn.functional as F
import torch.distributions as dist
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints import optimize_token
import torch.nn as nn

# now import weights and biases
import wandb

from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse


# from unsupervised_keypoints.optimize_token import init_random_noise


def collect_maps(
    controller,
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=512,
    layers=[0, 1, 2, 3, 4, 5],
    indices=None,
):
    """
    returns the bilinearly upsampled attention map of size upsample_res x upsample_res for the first word in the prompt
    """

    attention_maps = controller.get_average_attention()

    imgs = []

    layer_overall = -1

    for key in from_where:
        for layer in range(len(attention_maps[key])):
            layer_overall += 1

            if layer_overall not in layers:
                continue

            img = attention_maps[key][layer]

            img = img.reshape(
                4, int(img.shape[1] ** 0.5), int(img.shape[1] ** 0.5), img.shape[2]
            )

            if indices is not None:
                img = img[:, :, :, indices]

            img = img.permute(0, 3, 1, 2)

            if upsample_res != -1:
                # bilinearly upsample the image to img_sizeximg_size
                img = F.interpolate(
                    img,
                    size=(upsample_res, upsample_res),
                    mode="bilinear",
                    align_corners=False,
                )

            imgs.append(img)

    imgs = torch.stack(imgs, dim=0)

    return imgs


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


def equivariance_loss(
    embeddings_initial,
    embeddings_transformed,
    transform,
    kernel_size=5,
    sigma=1.0,
    temperature=1e-1,
    device="cuda",
):
    # get the argmax for both embeddings_initial and embeddings_uninverted

    initial_pos = eval.find_max_pixel(embeddings_initial) / 32
    transformed_pos = eval.find_max_pixel(embeddings_transformed) / 32

    transformed_pos_prime = transform.transform_keypoints(initial_pos)

    initial_pos_prime = transform.inverse_transform_keypoints(transformed_pos)

    within_image = ((transformed_pos_prime < 1) * (transformed_pos_prime > 0)).sum(
        dim=1
    ) == 2

    loss_initial = find_gaussian_loss_at_point(
        embeddings_initial,
        initial_pos_prime,
        sigma=sigma,
        temperature=temperature,
        device=device,
        indices=within_image,
    )

    loss_transformed = find_gaussian_loss_at_point(
        embeddings_transformed,
        transformed_pos_prime,
        sigma=sigma,
        temperature=temperature,
        device=device,
        indices=within_image,
    )

    return (loss_initial + loss_transformed) / torch.sum(within_image)


def sharpening_loss(attn_map, sigma=1.0, temperature=1e-1, device="cuda"):
    pos = eval.find_max_pixel(attn_map) / 32

    loss = find_gaussian_loss_at_point(
        attn_map,
        pos,
        sigma=sigma,
        temperature=temperature,
        device=device,
    )

    return loss


def find_gaussian_loss_at_point(
    attn_map, pos, sigma=1.0, temperature=1e-1, device="cuda", indices=None
):
    """
    pos is a location between 0 and 1
    """
    # attn_map is of shape (T, H, W)
    T, H, W = attn_map.shape

    # Scale attn_map by temperature
    attn_map_scaled = attn_map / temperature

    # Apply spatial softmax over attn_map to get probabilities
    spatial_softmax = torch.nn.Softmax2d()
    attn_probs = spatial_softmax(attn_map_scaled)

    # Create Gaussian circle at the given position
    target = optimize_token.gaussian_circle(
        pos, size=H, sigma=sigma, device=device
    )  # Assuming H and W are the same
    target = target.to(attn_map.device)

    # Normalize the target
    target = target / target.max()

    # possibly select a subset of indices
    if indices is not None:
        attn_probs = attn_probs[indices]
        target = target[indices]

    # Compute loss
    loss = F.mse_loss(attn_probs, target)

    return loss


def ddpm_loss(ldm, image, selected_context, masks, noise_level=-8, device="cuda"):
    """
    Passing in just the selected tokens, this masks the region that they can see (after detaching from the graph)
    This is a regularizer to make sure the token represents a similar concept across the dataset
    making sure the token uses the image information
    """

    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    with torch.no_grad():
        latent = ptp_utils.image2latent(ldm, image, device)

    noise = torch.rand_like(latent)

    noisy_image = ldm.scheduler.add_noise(
        latent, noise, ldm.scheduler.timesteps[noise_level]
    )

    controller = ptp_utils.AttentionStore()

    ptp_utils.register_attention_control(ldm, controller)

    noise_pred = ldm.unet(
        noisy_image,
        ldm.scheduler.timesteps[noise_level],
        encoder_hidden_states=selected_context,
    )["sample"]

    _mask = masks.reshape(masks.shape[0], masks.shape[1] * masks.shape[2]).detach()
    _mask = _mask / _mask.max(dim=1, keepdim=True)[0]
    _mask = _mask.sum(dim=0)
    _mask = _mask.reshape(1, 1, int(_mask.shape[0] ** 0.5), int(_mask.shape[0] ** 0.5))
    # bilinearly upsample to noise_pred.shape
    _mask = F.interpolate(
        _mask,
        size=(noise_pred.shape[2], noise_pred.shape[3]),
        mode="bilinear",
        align_corners=False,
    )

    ddpm_loss = nn.MSELoss()(noise_pred * _mask, noise * _mask) / torch.sum(_mask)

    return ddpm_loss


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


def optimize_embedding(
    ldm,
    wandb_log=True,
    context=None,
    device="cuda",
    num_steps=2000,
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    lr=5e-3,
    noise_level=-1,
    num_tokens=1000,
    top_k=10,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augment_shear=(0.0, 0.0),
    sdxl=False,
    mafl_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/TCDCN-face-alignment/MAFL/",
    celeba_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    sigma=1.0,
    equivariance_loss_weight=0.1,
):
    if wandb_log:
        # start a wandb session
        wandb.init(project="attention_maps")

    dataset = CelebA(split="train", mafl_loc=mafl_loc, celeba_loc=celeba_loc)

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
        shear=augment_shear,
    )

    # every iteration return image, pixel_loc

    if context is None:
        context = ptp_utils.init_random_noise(device, num_words=num_tokens)
        # context = (
        #     torch.load("proper_translation_in_augmentations/embedding.pt")
        #     .to(device)
        #     .detach()
        # )
        # context = torch.load("embedding.pt").to(device).detach()

    context.requires_grad = True

    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)

    # time the optimization
    import time

    start = time.time()

    prev_loss = 0
    prev_acc = 0

    for iteration in range(num_steps):
        index = np.random.randint(len(dataset))
        mini_batch = dataset[index]

        image = mini_batch["img"]

        attn_maps = ptp_utils.run_and_find_attn(
            ldm,
            image,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
            device=device,
        )

        transformed_img = invertible_transform(image)

        attention_maps_transformed = ptp_utils.run_and_find_attn(
            ldm,
            transformed_img,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
            device=device,
        )

        top_embedding_indices = ptp_utils.find_top_k(
            attn_maps.view(num_tokens, -1), top_k
        )

        # never had transformation applied
        best_embeddings = attn_maps[top_embedding_indices]
        # transformed image, then found attn maps
        best_embeddings_transformed = attention_maps_transformed[top_embedding_indices]

        # _ddpm_loss = (
        #     ddpm_loss(
        #         ldm,
        #         image,
        #         context[:, top_embedding_indices],
        #         best_embeddings,
        #         noise_level=noise_level,
        #         device=device,
        #     )
        #     * 1000
        # )
        _ddpm_loss = torch.tensor(0).to(device)

        _sharpening_loss = sharpening_loss(best_embeddings, device=device, sigma=sigma)

        _loss_equivariance = equivariance_loss(
            best_embeddings,
            best_embeddings_transformed,
            invertible_transform,
            sigma=sigma,
            device=device,
        )
        # instead get the argmax of each and apply it to the other
        # then bluring etc as in sharpening loss
        # _loss_equivariance = nn.MSELoss()(best_embeddings_vanilla, best_embeddings_uninverted) * 10

        loss = (
            _loss_equivariance * equivariance_loss_weight
            + _sharpening_loss
            + _ddpm_loss
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if wandb_log:
            wandb.log(
                {
                    "loss": loss.item(),
                    "_loss_equivariance": _loss_equivariance.item(),
                    "_sharpening_loss": _sharpening_loss.item(),
                    "_ddpm_loss": _ddpm_loss.item(),
                }
            )
        else:
            print(
                f"loss: {loss.item()}, _loss_equivariance: {_loss_equivariance.item()}, sharpening_loss: {_sharpening_loss.item()}"
            )

    print(f"optimization took {time.time() - start} seconds")

    return context


# def optimize_embedding_ddpm(
#     ldm,
#     wandb_log=True,
#     context=None,
#     device="cuda",
#     num_steps=2000,
#     from_where=["down_cross", "mid_cross", "up_cross"],
#     upsample_res=32,
#     layers=[0, 1, 2, 3, 4, 5],
#     lr=5e-3,
#     noise_level=-1,
#     num_tokens=1000,
#     top_k=10,
# ):
#     if wandb_log:
#         # start a wandb session
#         wandb.init(project="attention_maps")

#     dataset = CelebA()

#     context = ptp_utils.init_random_noise(device, num_words=num_tokens)

#     image_context = ptp_utils.init_random_noise(device, num_words=1)

#     context.requires_grad = True
#     image_context.requires_grad = True

#     # optimize context to maximize attention at pixel_loc
#     optimizer = torch.optim.Adam([context, image_context], lr=lr)

#     for _ in range(num_steps):
#         mini_batch = dataset[np.random.randint(len(dataset))]

#         image = mini_batch["img"]

#         # if image is a torch.tensor, convert to numpy
#         if type(image) == torch.Tensor:
#             image = image.permute(1, 2, 0).detach().cpu().numpy()

#         with torch.no_grad():
#             latent = ptp_utils.image2latent(ldm, image, device)

#         noise = torch.rand_like(latent)

#         noisy_image = ldm.scheduler.add_noise(
#             latent, noise, ldm.scheduler.timesteps[noise_level]
#         )

#         controller = ptp_utils.AttentionStore()

#         ptp_utils.register_attention_control(ldm, controller)
#         # sdxl_monkey_patch.register_attention_control(ldm, controller)

#         _, _ = ptp_utils.diffusion_step(
#             ldm,
#             controller,
#             noisy_image,
#             context,
#             ldm.scheduler.timesteps[noise_level],
#             cfg=False,
#         )

#         attention_maps = collect_maps(
#             controller,
#             from_where=from_where,
#             upsample_res=upsample_res,
#             layers=layers,
#             number_of_maps=num_tokens,
#         )

#         # take the mean over the first 2 dimensions
#         attention_maps = torch.mean(attention_maps, dim=(0, 1))

#         import torch.distributions as dist

#         # Normalize the activation maps to represent probability distributions
#         attention_maps_softmax = torch.softmax(
#             attention_maps.view(num_tokens, -1), dim=-1
#         )

#         # Compute the entropy of each token
#         entropy = dist.Categorical(
#             probs=attention_maps_softmax.view(num_tokens, -1)
#         ).entropy()

#         # Select the top_k tokens with the lowest entropy
#         _, top_embedding_indices = torch.topk(entropy, top_k, largest=False)

#         # Apply a gaussian loss to these embeddings
#         # otherwise there is no motivation to be sharp
#         best_embeddings = attention_maps[top_embedding_indices]
#         _gaussian_loss = gaussian_loss(best_embeddings)

#         this_context = context[:, top_embedding_indices]

#         # add tokens to be able to capture everything else not captured by the top_k tokens
#         this_context = torch.cat([this_context, image_context], dim=1)

#         # simply to avoid mismatch shape errors
#         controller = ptp_utils.AttentionStore()
#         ptp_utils.register_attention_control(ldm, controller)

#         noise_pred = ldm.unet(
#             noisy_image,
#             ldm.scheduler.timesteps[noise_level],
#             encoder_hidden_states=this_context,
#         )["sample"]

#         ddpm_loss = nn.MSELoss()(noise_pred, noise)

#         loss = _gaussian_loss * 1e3 + ddpm_loss

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if wandb_log:
#             wandb.log(
#                 {
#                     "loss": loss.item(),
#                     "gaussian_loss": _gaussian_loss.item(),
#                     "ddpm_loss": ddpm_loss.item(),
#                 }
#             )
#         else:
#             print(f"loss: {loss.item()}")

#     return context
