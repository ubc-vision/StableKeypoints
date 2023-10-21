import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.distributions as dist
from torch.optim.lr_scheduler import StepLR
from unsupervised_keypoints import ptp_utils
from unsupervised_keypoints.celeba import CelebA
from unsupervised_keypoints import cub
from unsupervised_keypoints import cub_parts
from unsupervised_keypoints import taichi
from unsupervised_keypoints import human36m
from unsupervised_keypoints import deepfashion
from unsupervised_keypoints.eval import pixel_from_weighted_avg, find_max_pixel
from unsupervised_keypoints.optimize import collect_maps
from unsupervised_keypoints.eval import (
    find_corresponding_points,
    run_image_with_context_augmented,
    # progressively_zoom_into_image,
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
    upsample_res=256,
    layers=[0, 1, 2, 3, 4, 5],
    from_where=["down_cross", "mid_cross", "up_cross"],
    num_tokens=1000,
    top_k=30,
    augment=False,
    augment_degrees=30,
    augment_scale=(0.9, 1.1),
    augment_translate=(0.1, 0.1),
    augment_shear=(0.0, 0.0),
    dataset_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    dataset_name = "celeba_aligned",
    min_dist = 0.05,
    controllers=None,
    num_gpus=1,
    top_k_strategy = "entropy",
    sigma = 3,
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
        dataset = human36m.TrainSet(data_root=dataset_loc, image_size=512)
    elif dataset_name == "deepfashion":
        dataset = deepfashion.TrainSet(data_root=dataset_loc, image_size=512)
    else:
        raise NotImplementedError

    invertible_transform = RandomAffineWithInverse(
        degrees=augment_degrees,
        scale=augment_scale,
        translate=augment_translate,
        shear=augment_shear,
    )

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

        if augment:
            image = invertible_transform(image)

        attention_maps = ptp_utils.run_and_find_attn(
            ldm,
            image,
            context,
            layers=layers,
            noise_level=noise_level,
            from_where=from_where,
            upsample_res=upsample_res,
            controllers=controllers,
        )
        
        for attention_map in attention_maps:
        
            if top_k_strategy == "entropy":
                top_embedding_indices = ptp_utils.find_top_k(
                    attention_map, top_k, min_dist = min_dist,
                )
            elif top_k_strategy == "gaussian":
                top_embedding_indices = ptp_utils.find_top_k_gaussian(
                    attention_map, top_k, min_dist = min_dist, sigma=sigma,
                )
            elif top_k_strategy == "consistent":
                top_embedding_indices = torch.arange(top_k)
        
            indices_list.append(top_embedding_indices.cpu())
    
    # find the top_k most common indices
    indices_list = torch.stack([index for index in indices_list])
    indices_list = indices_list.reshape(-1)
    indices, counts = torch.unique(indices_list, return_counts=True)
    indices = indices[counts.argsort(descending=True)]
    indices = indices[:top_k]

    return indices


def compose_transform(
    scale=(1.0, 1.0),
    translation=(0.0, 0.0),
    rotation=0.0,
    shear=(0.0, 0.0),
    center=(0.5, 0.5),
    device="cuda",
):
    # Convert rotation to radians
    theta = rotation * (3.14159 / 180.0)

    # Create individual transformation matrices
    T_scale = torch.tensor(
        [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]], dtype=torch.float32
    ).to(device)
    T_trans = torch.tensor(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]], dtype=torch.float32
    ).to(device)
    T_rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    ).to(device)
    T_shear = torch.tensor(
        [[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]], dtype=torch.float32
    ).to(device)

    # Transformation matrices for translating rotation center to origin and back
    T_center_to_origin = torch.tensor(
        [[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]], dtype=torch.float32
    ).to(device)
    T_origin_to_center = torch.tensor(
        [[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=torch.float32
    ).to(device)

    # Compose transformations
    T = torch.mm(
        T_trans,
        torch.mm(
            T_origin_to_center,
            torch.mm(T_shear, torch.mm(T_scale, torch.mm(T_rot, T_center_to_origin))),
        ),
    )

    return T


def transform_points(points, T):
    # Convert to homogeneous coordinates
    points_h = torch.cat(
        [points, torch.ones(points.shape[0], 1, dtype=points.dtype).to(points.device)],
        dim=1,
    )
    # Apply transformation
    transformed_points_h = torch.mm(points_h, T.t())
    # Convert back to 2D coordinates
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2].view(
        -1, 1
    )
    return transformed_points


def create_batch_of_augmentations(initial_keypoints, final_keypoints, device="cuda"):
    # Number of augmentations
    batch_size = 32

    # Initialize tensors to hold the batch of transformed keypoints
    transformed_initial_keypoints_batch = torch.zeros(
        (batch_size, *initial_keypoints.shape)
    ).to(device)
    transformed_final_keypoints_batch = torch.zeros(
        (batch_size, *final_keypoints.shape)
    ).to(device)

    transformed_initial_keypoints_batch[0] = initial_keypoints
    transformed_final_keypoints_batch[0] = final_keypoints

    # Generate a batch of differently rotated keypoints
    for i in range(1, batch_size):
        # create random scale, translation, rotation, and shear
        scale = ptp_utils.random_range(2, 0.95, 1.05)
        translation = ptp_utils.random_range(2, -0.05, 0.05)
        rotation = ptp_utils.random_range(1, -10, 10)
        shear = ptp_utils.random_range(2, -0.05, 0.05)

        # Create rotation matrix
        T_rot = compose_transform(
            scale=scale,
            translation=translation,
            rotation=rotation,
            shear=shear,
            device=device,
        )

        # Transform the keypoints
        transformed_initial_keypoints = transform_points(initial_keypoints, T_rot)
        transformed_final_keypoints = transform_points(final_keypoints, T_rot)

        # Store in batch tensors
        transformed_initial_keypoints_batch[i] = transformed_initial_keypoints
        transformed_final_keypoints_batch[i] = transformed_final_keypoints

    return transformed_initial_keypoints_batch, transformed_final_keypoints_batch



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
    augment_shear=(0.0, 0.0),
    augmentation_iterations=20,
    dataset_loc="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    visualize=False,
    dataset_name = "celeba_aligned",
    controllers=None,
    num_gpus=1,
    max_num_points = 50_000,
    max_loc_strategy="argmax",
    save_folder="outputs",
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
        dataset = human36m.TrainRegSet(data_root=dataset_loc, image_size=512)
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
            augment_shear=augment_shear,
            controllers=controllers,
            save_folder=save_folder,
            num_gpus=num_gpus,
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

    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression(fit_intercept=False)
    ransac = RANSACRegressor(
        base_estimator=linear_model,
        min_samples=int(0.1 * len(X)),  # 10% of the data
        max_trials=10000,  # Very high number of trials
        residual_threshold=None,  # Will set based on preliminary fit or other criteria
        loss="squared_error",
    )

    ransac.fit(X, Y)

    linear_model_fitted = ransac.estimator_

    # Get the coefficients and intercept
    W = linear_model_fitted.coef_

    return W.T


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    initial_keypoints = torch.rand(10, 2) / 2 + 0.25
    final_keypoints = torch.rand(5, 2) / 2 + 0.25

    batch_initial_keypoints, batch_final_keypoints = create_batch_of_augmentations(
        initial_keypoints, final_keypoints
    )

    # visualize the keypoints before and after
    fig, axs = plt.subplots(1, 10)
    for i in range(10):
        axs[i].scatter(
            batch_initial_keypoints[i, :, 0],
            batch_initial_keypoints[i, :, 1],
            marker="x",
            color="red",
        )
        axs[i].scatter(
            batch_final_keypoints[i, :, 0],
            batch_final_keypoints[i, :, 1],
            marker="x",
            color="blue",
        )

        # make the range of the axes between 0 and 1
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 1)

        axs[i].set_aspect("equal")
    # increase resolution of plot to 512x512
    fig.set_size_inches(2048 / 100, 2048 / 100)

    plt.savefig(f"outputs/rotated_points.png")
