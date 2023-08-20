# load the dataset
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints.celeba import CelebA

# now import weights and biases
import wandb

# from unsupervised_keypoints.optimize_token import init_random_noise

from unsupervised_keypoints.optimize import collect_maps


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


def get_attn_map(
    ldm,
    image,
    context,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    noise_level=-1,
    num_tokens=77,
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    with torch.no_grad():
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

    attention_maps = torch.mean(attention_maps, dim=(0, 1))

    return attention_maps


def run_example(
    ldm,
    batch,
    context,
    threshold_value=0.5,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    noise_level=-1,
):
    image = batch["img"]
    label = batch["label"]

    attention_maps = get_attn_map(
        ldm,
        image,
        context,
        device=device,
        from_where=from_where,
        upsample_res=upsample_res,
        layers=layers,
        noise_level=noise_level,
    )
    num_maps = attention_maps.shape[0]

    # the label should just be the maximum of the attention maps
    attn_map_max = torch.max(attention_maps)

    # compare threshold_value to attn_map_max with label
    # if below threshold_value, then label should be 0, if above threshold_value, then label should be 1
    est_label = (attn_map_max > threshold_value).to(torch.float)

    bool_correct = (est_label == label).to(torch.float)

    return bool_correct, attn_map_max


def eval_embedding(
    ldm,
    context,
    device="cuda",
    num_steps=2000,
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=32,
    layers=[0, 1, 2, 3, 4, 5],
    lr=5e-3,
    noise_level=-1,
):
    dogs = Cats_Dogs(dogs=True, train=False)
    cats = Cats_Dogs(dogs=False, train=False)

    if context is None:
        context = ptp_utils.init_random_noise(device)

    context.requires_grad = True

    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)

    dog_accuracies = []
    est_dog_maxes = []

    for i in tqdm(range(len(dogs))):
        batch = dogs[i]
        accuracy, est_max = run_example(ldm, batch, context, threshold_value=0.1)
        dog_accuracies.append(accuracy.cpu().numpy())
        est_dog_maxes.append(est_max.detach().cpu().numpy())

    cat_accuracies = []
    est_cat_maxes = []
    for i in tqdm(range(len(cats))):
        batch = cats[i]
        accuracy, est_max = run_example(ldm, batch, context, threshold_value=0.1)
        cat_accuracies.append(accuracy.cpu().numpy())
        est_cat_maxes.append(est_max.detach().cpu().numpy())

    print(f"dog accuracy: {np.mean(dog_accuracies)}")
    print(f"cat accuracy: {np.mean(cat_accuracies)}")
    print(f"overall accuracy: {np.mean(dog_accuracies + cat_accuracies)}")

    # save the bins for the dog and cat maxes with matplotlib
    import matplotlib.pyplot as plt

    plt.hist(est_dog_maxes, bins=100, alpha=0.5, label="dog")
    plt.hist(est_cat_maxes, bins=100, alpha=0.5, label="cat")
    plt.legend(loc="upper right")
    plt.show()
    plt.savefig("dog_cat_maxes.png")

    return np.mean(dog_accuracies + cat_accuracies)


def find_max_pixel(map):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]
    """

    batch_size, h, w = map.shape

    map = map.view(batch_size, -1)

    max_indices = torch.argmax(map, dim=-1)

    max_indices = max_indices.view(batch_size, 1)

    max_indices = torch.cat([max_indices // w, max_indices % w], dim=-1)

    # offset by a half a pixel to get the center of the pixel
    max_indices = max_indices + 0.5

    return max_indices


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


def crop_image(image, crop_percent=90):
    """pixel is an integer between 0 and image.shape[1] or image.shape[2]"""

    assert 0 < crop_percent <= 100, "crop_percent should be between 0 and 100"

    height, width, channels = image.shape
    crop_height = int(height * crop_percent / 100)
    crop_width = int(width * crop_percent / 100)

    x_start_max = width - crop_width
    y_start_max = height - crop_height

    # Choose a random top-left corner within the allowed bounds
    x_start = torch.randint(0, int(x_start_max) + 1, (1,)).item()
    y_start = torch.randint(0, int(y_start_max) + 1, (1,)).item()

    # Crop the image
    cropped_image = image[
        y_start : y_start + crop_height, x_start : x_start + crop_width
    ]

    # bilinearly upsample to 512x512
    cropped_image = torch.nn.functional.interpolate(
        torch.tensor(cropped_image[None]).permute(0, 3, 1, 2),
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )[0]

    return (
        cropped_image.permute(1, 2, 0).numpy(),
        y_start,
        crop_height,
        x_start,
        crop_width,
    )


@torch.no_grad()
def run_image_with_tokens_cropped(
    ldm,
    image,
    tokens,
    indices,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    num_iterations=20,
    noise_level=-1,
    crop_percent=90.0,
    image_mask=None,
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    num_samples = torch.zeros(len(layers), 4, len(indices), 512, 512).cuda()
    sum_samples = torch.zeros(len(layers), 4, len(indices), 512, 512).cuda()

    side_margin = int(512 * crop_percent / 100.0 * 0.1)

    for i in range(num_iterations):
        cropped_image, y_start, height, x_start, width = crop_image(
            image, crop_percent=crop_percent
        )

        latents = ptp_utils.image2latent(ldm, cropped_image, device)

        controller = ptp_utils.AttentionStore()

        ptp_utils.register_attention_control(ldm, controller)

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

        assert height == width

        _attention_maps = collect_maps(
            controller,
            from_where=from_where,
            upsample_res=height,
            layers=layers,
            indices=indices,
        )

        num_samples[
            :,
            :,
            :,
            y_start + side_margin : y_start + height - side_margin,
            x_start + side_margin : x_start + width - side_margin,
        ] += 1
        sum_samples[
            :,
            :,
            :,
            y_start + side_margin : y_start + height - side_margin,
            x_start + side_margin : x_start + width - side_margin,
        ] += _attention_maps[
            :, :, :, side_margin:-side_margin, side_margin:-side_margin
        ]

        _attention_maps = sum_samples / num_samples

        if image_mask is not None:
            _attention_maps = _attention_maps * image_mask[None, None].to(device)

    # visualize sum_samples/num_samples
    attention_maps = sum_samples / num_samples

    attention_maps[attention_maps != attention_maps] = 0

    if image_mask is not None:
        attention_maps = attention_maps * image_mask[None, None].to(device)

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
    crop_percent=90.0,
):
    dataset = CelebA(split="test")

    distances = []

    for i in range(len(dataset)):
        batch = dataset[i]

        img = batch["img"]

        attention_maps = run_image_with_tokens_cropped(
            ldm,
            img,
            context,
            indices,
            device=device,
            from_where=from_where,
            layers=layers,
            noise_level=noise_level,
            crop_percent=crop_percent,
        )

        attention_maps = torch.mean(attention_maps, dim=(0, 1))

        # get the argmax of each of the best_embeddings
        highest_indices = find_max_pixel(attention_maps) / 512.0

        # for i in range(attention_maps.shape[0]):
        #     save_img(attention_maps[i], img, highest_indices[i], f"attn_maps_{i:02d}")
        # pass

        estimated_kpts = regressor(highest_indices.view(-1))

        estimated_kpts = estimated_kpts.view(-1, 2)

        gt_kpts = batch["kpts"].cuda()

        # get l2 distance between estimated and gt kpts
        l2 = torch.sqrt(torch.sum((estimated_kpts - gt_kpts) ** 2, dim=-1))

        eye_dist = torch.sqrt(torch.sum((gt_kpts[0] - gt_kpts[1]) ** 2, dim=-1))

        l2 = l2 / eye_dist

        distances.append(l2.cpu())

        # if i % 100 == 0:
        print(
            f"{(i/len(dataset)):06f}: {i} mean distance: {torch.mean(torch.stack(distances))}",
            end="\r",
        )
