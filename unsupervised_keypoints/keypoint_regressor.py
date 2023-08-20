import torch
import wandb
import numpy as np
import torch.distributions as dist
from torch.optim.lr_scheduler import StepLR
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints.eval import find_max_pixel
from unsupervised_keypoints.optimize import collect_maps
from unsupervised_keypoints.eval import (
    find_corresponding_points,
    run_image_with_tokens_cropped,
)

from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse

# from unsupervised_keypoints.optimize_token import (
#     init_random_noise,
#     # image2latent,
#     # AttentionStore,
# )


class LinearProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjection, self).__init__()
        self.mlp = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        y = self.mlp(x)
        return y


@torch.no_grad()
def find_best_indices(
    ldm,
    context,
    num_steps=100,
    device="cuda",
    noise_level=-1,
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    from_where=["down_cross", "mid_cross", "up_cross"],
    num_tokens=1000,
    top_k=30,
    augment=False,
):
    # TODO if augment then use the invertable warp
    dataset = CelebA(split="train")

    invertible_transform = RandomAffineWithInverse(
        degrees=30, scale=(1.0, 1.1), translate=(0.1, 0.1)
    )

    maps = []

    for _ in range(num_steps):
        mini_batch = dataset[np.random.randint(len(dataset))]

        image = mini_batch["img"]

        if augment:
            image = invertible_transform(image)

        # label = mini_batch['label']

        # if image is a torch.tensor, convert to numpy
        if type(image) == torch.Tensor:
            image = image.permute(1, 2, 0).detach().cpu().numpy()

        latent = ptp_utils.image2latent(ldm, image, device)

        noisy_image = ldm.scheduler.add_noise(
            latent, torch.rand_like(latent), ldm.scheduler.timesteps[noise_level]
        )

        controller = ptp_utils.AttentionStore()

        ptp_utils.register_attention_control(ldm, controller)

        _ = ptp_utils.diffusion_step(
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
        )

        # take the mean over the first 2 dimensions
        attention_maps = torch.mean(attention_maps, dim=(0, 1))

        maps.append(attention_maps)

    maps = torch.stack(maps, dim=0)

    maps = maps.reshape(num_steps, num_tokens, upsample_res, upsample_res)

    points, indices = find_corresponding_points(maps, num_points=top_k)

    return indices


def supervise_regressor(
    ldm,
    context,
    top_indices,
    wandb_log=True,
    lr=1e-3,
    anneal_after_num_steps=1e3,
    num_steps=1e4,
    device="cuda",
    noise_level=-1,
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    from_where=["down_cross", "mid_cross", "up_cross"],
    num_tokens=1000,
    top_k=30,
):
    if wandb_log:
        # start a wandb session
        wandb.init(project="regressor")

    dataset = CelebA()

    regressor = LinearProjection(
        input_dim=top_k * 2, output_dim=dataset.num_kps * 2
    ).cuda()

    # supervise the regressor parameters
    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)

    for iteration in range(int(num_steps)):
        mini_batch = dataset[np.random.randint(len(dataset))]

        image = mini_batch["img"]

        # label = mini_batch['label']

        # if image is a torch.tensor, convert to numpy
        if type(image) == torch.Tensor:
            image = image.permute(1, 2, 0).detach().cpu().numpy()

        with torch.no_grad():
            attention_maps = run_image_with_tokens_cropped(
                ldm,
                image,
                context,
                top_indices,
                device=device,
                from_where=from_where,
                layers=layers,
                noise_level=noise_level,
                crop_percent=90.0,
            )

            attention_maps = torch.mean(attention_maps, dim=(0, 1))

            # get the argmax of each of the best_embeddings
            highest_indices = find_max_pixel(attention_maps) / 512.0

        estimated_kpts = regressor(highest_indices.view(-1))

        estimated_kpts = estimated_kpts.view(-1, 2)

        gt_kpts = mini_batch["kpts"].cuda()

        loss = torch.nn.functional.mse_loss(estimated_kpts, gt_kpts)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Check if it's step 1000, and if so, modify the learning rate
        if iteration == int(anneal_after_num_steps):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * 0.1

        if wandb_log:
            wandb.log({"loss": loss.item()})
        else:
            print(f"loss: {loss.item()}")

    return regressor
