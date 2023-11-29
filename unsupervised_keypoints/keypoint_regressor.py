import torch
from tqdm import tqdm
from unsupervised_keypoints import ptp_utils
from datasets.celeba import CelebA
from datasets import custom_images
from datasets import cub
from datasets import cub_parts
from datasets import taichi
from datasets import human36m
from datasets import unaligned_human36m
from datasets import deepfashion
from unsupervised_keypoints.eval import pixel_from_weighted_avg, find_max_pixel
from unsupervised_keypoints.eval import run_image_with_context_augmented


@torch.no_grad()
def find_best_indices(
    ldm,
    context,
    num_steps=100,
    device="cuda",
    noise_level=-1,
    upsample_res=256,
    layers=[0, 1, 2, 3, 4, 5],
    from_where=["down_cross", "mid_cross", "up_cross"],
    num_tokens=1000,
    top_k=30,
    dataset_loc="~",
    dataset_name = "celeba_aligned",
    min_dist = 0.05,
    furthest_point_num_samples=50,
    controllers=None,
    num_gpus=1,
    top_k_strategy = "entropy",
    sigma = 3,
    validation = False,
    num_subjects=1,
):
    if dataset_name == "celeba_aligned":
        dataset = CelebA(split="train", dataset_loc=dataset_loc)
    elif dataset_name == "celeba_wild":
        dataset = CelebA(split="train", dataset_loc=dataset_loc, align = False)
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

    maps = []
    indices_list = []

    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_gpus, shuffle=True, drop_last=True)

    dataloader_iter = iter(dataloader)

    for _ in tqdm(range(num_steps//num_gpus)):

        try:
            mini_batch = next(dataloader_iter)
        except StopIteration:  # Explicitly catch StopIteration
            dataloader_iter = iter(dataloader)
            mini_batch = next(dataloader_iter)

        image = mini_batch["img"]

        attention_maps = ptp_utils.run_and_find_attn(
            ldm,
            image,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
            controllers=controllers,
            human36m = dataset_name == "human3.6m",
        )
        
        for attention_map in attention_maps:
        
            if top_k_strategy == "entropy":
                top_initial_candidates = ptp_utils.entropy_sort(
                    attention_map, furthest_point_num_samples, 
                )
            elif top_k_strategy == "gaussian":
                top_initial_candidates = ptp_utils.find_top_k_gaussian(
                    attention_map, furthest_point_num_samples, sigma=sigma, num_subjects = num_subjects
                )
            elif top_k_strategy == "consistent":
                top_initial_candidates = torch.arange(furthest_point_num_samples)
            else:
                raise NotImplementedError
            
            top_embedding_indices = ptp_utils.furthest_point_sampling(attention_map, top_k, top_initial_candidates)
        
            indices_list.append(top_embedding_indices.cpu())
    
    # find the top_k most common indices
    indices_list = torch.cat([index for index in indices_list])
    # indices_list = indices_list.reshape(-1)
    indices, counts = torch.unique(indices_list, return_counts=True)
    indices = indices[counts.argsort(descending=True)]
    indices = indices[:top_k]

    return indices


@torch.no_grad()
def precompute_all_keypoints(
    ldm,
    context,
    top_indices,
    device="cuda",
    noise_level=-1,
    layers=[0, 1, 2, 3, 4, 5],
    from_where=["down_cross", "mid_cross", "up_cross"],
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augmentation_iterations=20,
    dataset_loc="~",
    visualize=False,
    dataset_name = "celeba_aligned",
    controllers=None,
    num_gpus=1,
    max_num_points = 50_000,
    max_loc_strategy="argmax",
    save_folder="outputs",
    validation = False,
):
    if dataset_name == "celeba_aligned":
        dataset = CelebA(split="train", dataset_loc=dataset_loc)
    elif dataset_name == "celeba_wild":
        dataset = CelebA(split="train", dataset_loc=dataset_loc, align = False)
    elif dataset_name == "cub_aligned":
        dataset = cub.TrainRegSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "cub_001":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train", single_class=1)
    elif dataset_name == "cub_002":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train", single_class=2)
    elif dataset_name == "cub_003":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train", single_class=3)
    elif dataset_name == "cub_all":
        dataset = cub_parts.CUBDataset(dataset_root=dataset_loc, split="train")
    elif dataset_name == "taichi":
        dataset = taichi.TrainRegSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "human3.6m":
        dataset = human36m.TrainRegSet(data_root=dataset_loc, validation=validation)
    elif dataset_name == "unaligned_human3.6m":
        dataset = unaligned_human36m.TrainRegSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "deepfashion":
        dataset = deepfashion.TrainRegSet(data_root=dataset_loc, image_size=512)
    else:
        raise NotImplementedError

    source_keypoints = []
    target_keypoints = []
    visibility = []

    # create dataloader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    dataloader_iter = iter(dataloader)

    for _ in tqdm(range(min(len(dataset), max_num_points))):

        mini_batch = next(dataloader_iter)


        image = mini_batch["img"][0]
        kpts = mini_batch["kpts"][0]
        
        
        target_keypoints.append(kpts)
        
    
        if "visibility" in mini_batch:
            visibility.append(mini_batch["visibility"][0])

        # if image is a torch.tensor, convert to numpy
        if type(image) == torch.Tensor:
            image = image.permute(1, 2, 0).detach().cpu().numpy()

        attention_maps = run_image_with_context_augmented(
            ldm,
            image,
            context,
            top_indices,
            device=device,
            from_where=from_where,
            layers=layers,
            noise_level=noise_level,
            augmentation_iterations=augmentation_iterations,
            augment_degrees=augment_degrees,
            augment_scale=augment_scale,
            augment_translate=augment_translate,
            controllers=controllers,
            save_folder=save_folder,
            num_gpus=num_gpus,
            human36m = dataset_name == "human3.6m",
        )
        if max_loc_strategy == "argmax":
            highest_indices = find_max_pixel(attention_maps) / 512.0
        else:
            highest_indices = pixel_from_weighted_avg(attention_maps) / 512.0

        source_keypoints.append(highest_indices)

    return torch.stack(source_keypoints), torch.stack(target_keypoints), torch.stack(visibility) if len(visibility) > 0 else None


def return_regressor_visible(X, Y, visible):
    import numpy as np
    
    # find mean of X
    X = X - 0.5
    Y = Y - 0.5

    # Initialize W to have the same number of columns as keypoints
    W = np.zeros((X.shape[1], Y.shape[1]))

    # Iterate through each keypoint
    for j in range(Y.shape[1]):
        # Indices where this keypoint is visible
        visible_indices = np.where(visible[:, j] == 1)[0]
        
        # Filter X and Y matrices based on visibility of this keypoint
        X_filtered = X[visible_indices, :]
        Y_filtered = Y[visible_indices, j]

        # Solve for the weights related to this keypoint
        W_j = np.linalg.pinv(X_filtered.T @ X_filtered) @ X_filtered.T @ Y_filtered
        
        # Store these weights in the W matrix
        W[:, j] = W_j

    return W


def return_regressor(X, Y):
    import numpy as np
    
    # find mean of X
    X = X - 0.5
    Y = Y - 0.5

    # # W = np.linalg.inv(X.T @ X) @ X.T @ Y
    W = np.linalg.pinv(X.T @ X) @ X.T @ Y

    return W


def return_regressor_human36m(X, Y):
    
    from unsupervised_keypoints.eval import swap_points
    
    import numpy as np
    
    X = torch.tensor(X)-0.5
    Y = torch.tensor(Y)-0.5
    
    XTXXT = (X.T @ X).inverse() @ X.T
    
    while True:
        W = XTXXT @ Y
        pred_y = X @ W
        
        pred_y = torch.tensor(pred_y)

        dist = (pred_y - Y).reshape(X.shape[0], -1, 2).norm(dim=2).mean(dim=1)

        swaped_y = swap_points(Y.reshape(Y.shape[0], -1, 2)).reshape(Y.shape[0], -1)
        swaped_dist = (pred_y - swaped_y).reshape(X.shape[0], -1, 2).norm(dim=2).mean(dim=1)

        should_swap = dist > swaped_dist

        if should_swap.sum() > 10:
            print("should swap sum, ", should_swap.sum())
            Y[should_swap] = swaped_y[should_swap]
        else:
            break
    

    return W.numpy()

