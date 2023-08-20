# load the dataset
import torch
import numpy as np
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints import sdxl_monkey_patch
import torch.nn.functional as F
import torch.distributions as dist
from unsupervised_keypoints.celeba import CelebA
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


class GaussianLoss(nn.Module):
    def __init__(self, T, buffer_size=10, device="cuda", threshold=0.02):
        super().__init__()
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.token_activation_history = torch.zeros(T, buffer_size).to(device)

    def forward(self, attn_map, temperature=1e-4):
        # attn_map is of shape (T, H, W)
        T, H, W = attn_map.shape

        # Softmax over flattened attn_map to get probabilities
        attn_probs = F.softmax(
            attn_map.view(T, -1) / temperature, dim=1
        )  # Shape: (T, H*W)

        # Stop the gradient for attn_probs
        attn_probs = attn_probs.detach()

        # Find the max of attn_probs for each token
        max_values = torch.max(attn_probs.view(T, -1), dim=1)[0]

        # Shift token activation history and append new max_values
        self.token_activation_history = torch.roll(
            self.token_activation_history, shifts=-1, dims=1
        )
        self.token_activation_history[:, -1] = max_values

        # Compute maximum activation over history
        max_history_values = self.token_activation_history.max(dim=1)[0]

        # Divide attn_probs by the max of the first dim, weighted by the token weights
        attn_probs = attn_probs / max_history_values[:, None]
        attn_probs = attn_probs.view(T, H, W)  # Reshape back to original shape

        max_attn_map = torch.max(attn_map.view(T, -1), dim=1)[0]
        # Apply thresholding: suppress entries below the threshold, keep the attn_probs otherwise
        attn_probs = torch.where(
            max_attn_map[:, None, None] < self.threshold,
            torch.zeros_like(attn_probs),
            attn_probs,
        )

        loss = F.mse_loss(attn_map, attn_probs)

        return loss


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
    sdxl=False,
):
    if wandb_log:
        # start a wandb session
        wandb.init(project="attention_maps")

    dataset = CelebA(split="train")

    invertible_transform = RandomAffineWithInverse(
        degrees=30, scale=(1.0, 1.1), translate=(0.1, 0.1)
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

    prev_loss = 0
    prev_acc = 0

    loss_fn = GaussianLoss(T=num_tokens)

    for iteration in range(num_steps):
        index = np.random.randint(len(dataset))
        mini_batch = dataset[index]

        image = mini_batch["img"]

        with torch.no_grad():
            vanilla_attn_maps = ptp_utils.run_and_find_attn(
                ldm,
                image,
                context,
                layers=layers,
                noise_level=noise_level,
                from_where=from_where,
                upsample_res=upsample_res,
            )

        transformed_img = invertible_transform(image)

        attention_maps = ptp_utils.run_and_find_attn(
            ldm,
            transformed_img,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
        )

        uninverted_attn_maps = invertible_transform.inverse(attention_maps)

        # Normalize the activation maps to represent probability distributions
        attention_maps_softmax = torch.softmax(
            attention_maps.view(num_tokens, -1), dim=-1
        )

        # Compute the entropy of each token
        entropy = dist.Categorical(
            probs=attention_maps_softmax.view(num_tokens, -1)
        ).entropy()

        # Select the top_k tokens with the lowest entropy
        _, top_embedding_indices = torch.topk(entropy, top_k, largest=False)

        # transformed image, then found attn maps
        best_embeddings = attention_maps[top_embedding_indices]
        # transformed, found attn maps, then inverted
        best_embeddings_uninverted = uninverted_attn_maps[top_embedding_indices]
        # never had transformation applied
        best_embeddings_vanilla = vanilla_attn_maps[top_embedding_indices]

        _gaussian_loss = gaussian_loss(best_embeddings)
        l2_loss = nn.MSELoss()(best_embeddings_vanilla, best_embeddings_uninverted)
        loss = l2_loss + _gaussian_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if wandb_log:
            wandb.log(
                {
                    "loss": loss.item(),
                    "l2_loss": l2_loss.item(),
                    "_gaussian_loss": _gaussian_loss.item(),
                }
            )
        else:
            print(
                f"loss: {loss.item()}, l2_loss: {l2_loss.item()}, gaussian_loss: {gaussian_loss.item()}"
            )

    print(f"optimization took {time.time() - start} seconds")

    return context


def optimize_embedding_ddpm(
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
):
    if wandb_log:
        # start a wandb session
        wandb.init(project="attention_maps")

    dataset = CelebA()

    context = ptp_utils.init_random_noise(device, num_words=num_tokens)

    image_context = ptp_utils.init_random_noise(device, num_words=1)

    context.requires_grad = True
    image_context.requires_grad = True

    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context, image_context], lr=lr)

    for _ in range(num_steps):
        mini_batch = dataset[np.random.randint(len(dataset))]

        image = mini_batch["img"]

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
        # sdxl_monkey_patch.register_attention_control(ldm, controller)

        _, _ = ptp_utils.diffusion_step(
            ldm,
            controller,
            noisy_image,
            context,
            ldm.scheduler.timesteps[noise_level],
            cfg=False,
        )

        attention_maps = collect_maps(
            controller,
            from_where=from_where,
            upsample_res=upsample_res,
            layers=layers,
            number_of_maps=num_tokens,
        )

        # take the mean over the first 2 dimensions
        attention_maps = torch.mean(attention_maps, dim=(0, 1))

        import torch.distributions as dist

        # Normalize the activation maps to represent probability distributions
        attention_maps_softmax = torch.softmax(
            attention_maps.view(num_tokens, -1), dim=-1
        )

        # Compute the entropy of each token
        entropy = dist.Categorical(
            probs=attention_maps_softmax.view(num_tokens, -1)
        ).entropy()

        # Select the top_k tokens with the lowest entropy
        _, top_embedding_indices = torch.topk(entropy, top_k, largest=False)

        # Apply a gaussian loss to these embeddings
        # otherwise there is no motivation to be sharp
        best_embeddings = attention_maps[top_embedding_indices]
        _gaussian_loss = gaussian_loss(best_embeddings)

        this_context = context[:, top_embedding_indices]

        # add tokens to be able to capture everything else not captured by the top_k tokens
        this_context = torch.cat([this_context, image_context], dim=1)

        # simply to avoid mismatch shape errors
        controller = ptp_utils.AttentionStore()
        ptp_utils.register_attention_control(ldm, controller)

        noise_pred = ldm.unet(
            noisy_image,
            ldm.scheduler.timesteps[noise_level],
            encoder_hidden_states=this_context,
        )["sample"]

        ddpm_loss = nn.MSELoss()(noise_pred, noise)

        loss = _gaussian_loss * 1e3 + ddpm_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if wandb_log:
            wandb.log(
                {
                    "loss": loss.item(),
                    "gaussian_loss": _gaussian_loss.item(),
                    "ddpm_loss": ddpm_loss.item(),
                }
            )
        else:
            print(f"loss: {loss.item()}")

    return context
