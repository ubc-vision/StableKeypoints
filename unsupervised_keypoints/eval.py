# load the dataset
import torch
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
import torch.nn.functional as F
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints.invertable_transform import RandomAffineWithInverse

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


def find_max_pixel(map, return_confidences=False):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]
    """

    batch_size, h, w = map.shape

    map_reshaped = map.view(batch_size, -1)

    max_indices = torch.argmax(map_reshaped, dim=-1)

    max_indices = max_indices.view(batch_size, 1)

    max_indices = torch.cat([max_indices // w, max_indices % w], dim=-1)

    # offset by a half a pixel to get the center of the pixel
    max_indices = max_indices + 0.5

    if not return_confidences:
        return max_indices

    batch_indices = torch.arange(10, device="cuda:0").view(-1, 1)
    indices = torch.cat((batch_indices, max_indices), dim=1).long()

    # Use the indices to gather the values
    values = map[indices[:, 0], indices[:, 1], indices[:, 2]]

    return max_indices, values


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
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    num_samples = torch.zeros(len(indices), 512, 512).cuda()
    sum_samples = torch.zeros(len(indices), 512, 512).cuda()

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
        shear=augment_shear,
    )

    if visualize:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(augmentation_iterations + 1, 5)

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

        _attention_maps = collect_maps(
            controller,
            from_where=from_where,
            upsample_res=512,
            layers=layers,
            indices=indices,
        )

        _attention_maps = _attention_maps.mean((0, 1))

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
):
    dataset = CelebA(split="test")

    distances = []

    eye_dists = []

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
            # visualize=True,
        )
        # attention_maps = run_image_with_tokens_augmented(
        #     ldm,
        #     img,
        #     context,
        #     indices,
        #     device=device,
        #     from_where=from_where,
        #     layers=layers,
        #     noise_level=noise_level,
        #     augmentation_iterations=1,
        #     augment_degrees=0,
        #     augment_scale=(1.0, 1.0),
        #     augment_translate=(0.0, 0.0),
        #     augment_shear=(0.0, 0.0),
        #     # visualize=True,
        # )

        # get the argmax of each of the best_embeddings
        highest_indices, confidences = find_max_pixel(
            attention_maps, return_confidences=True
        )
        highest_indices = highest_indices / 512.0

        # for i in range(attention_maps.shape[0]):
        #     save_img(attention_maps[i], img, highest_indices[i], f"attn_maps_{i:02d}")
        # pass

        # estimated_kpts = regressor(highest_indices.view(-1))
        estimated_kpts = highest_indices.view(1, -1) @ regressor

        estimated_kpts = estimated_kpts.view(-1, 2)

        gt_kpts = batch["kpts"].cuda()

        # get l2 distance between estimated and gt kpts
        l2 = torch.sqrt(torch.sum((estimated_kpts - gt_kpts) ** 2, dim=-1))

        eye_dist = torch.sqrt(torch.sum((gt_kpts[0] - gt_kpts[1]) ** 2, dim=-1))

        l2 = l2 / eye_dist

        l2_mean = torch.mean(l2)

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
        eye_dists.append(eye_dist.cpu())

        print(
            f"{(i/len(dataset)):06f}: {i} mean distance: {torch.mean(torch.stack(distances))}, per keypoint: {torch.mean(torch.stack(distances), dim=0)}, eye_dist: {torch.mean(torch.stack(eye_dists))}",
            end="\r",
        )

        if i % 100 == 0:
            print()
        # Extract the 10 worst distances (and their indices) from the priority queue

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
    torch.save(torch.tensor(all_values), "argsort_test.pt")
