import os
import wandb
import numpy as np
import argparse
import torch
import numpy as np
from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.optimize import optimize_embedding

from unsupervised_keypoints.keypoint_regressor import (
    find_best_indices,
    precompute_all_keypoints,
    return_regressor,
    return_regressor_visible,
    return_regressor_human36m,
)

from unsupervised_keypoints.eval import evaluate
from unsupervised_keypoints.visualize import visualize_attn_maps, create_vid


# Argument parsing
parser = argparse.ArgumentParser(description="optimize a class embedding")

# Network details
parser.add_argument(
    "--model_type",
    type=str,
    default="sd-legacy/stable-diffusion-v1-5",
    help="ldm model type",
)
parser.add_argument(
    "--my_token",
    type=str,
    required=True,
    help="Hugging Face token for model download. Create a read token from https://huggingface.co/settings/tokens",
)
# Dataset details
parser.add_argument(
    "--dataset_loc",
    type=str,
    default="~",
    help="Path to dataset",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="outputs",
    help="Where to save visualizations and checkpoints",
)
parser.add_argument(
    "--wandb_name",
    type=str,
    default="temp",
    help="name of the wandb run",
)
parser.add_argument(
    "--dataset_name",
    # set the choices to be "mafl" and "celeba_aligned"
    choices=["celeba_aligned", "celeba_wild", "cub_aligned", "cub_001", "cub_002", "cub_003", "cub_all", "deepfashion", "taichi", "human3.6m", "unaligned_human3.6m", "custom"],
    type=str,
    default="celeba_aligned",
    help="name of the dataset to use",
)
parser.add_argument(
    "--max_len",
    type=int,
    default=-1,
    help="max length of the dataset. -1 means no max length",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
parser.add_argument("--wandb", action="store_true", help="wandb logging")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
parser.add_argument(
    "--num_steps", type=int, default=500, help="number of steps to optimize for"
)
parser.add_argument(
    "--num_tokens", type=int, default=500, help="number of tokens to optimize"
)
parser.add_argument(
    "--feature_upsample_res", type=int, default=128, help="upsampled resolution for latent features grabbed from the attn operation"
)
parser.add_argument(
    "--batch_size", type=int, default=4, help="size of the batch for optimization"
)
parser.add_argument(
    "--top_k_strategy",
    type=str,
    default="gaussian",
    choices=["entropy", "gaussian", "consistent"],
    help="strategy for choosing top k tokens",
)
parser.add_argument(
    "--max_loc_strategy",
    type=str,
    default="argmax",
    choices=["argmax", "weighted_avg"],
    help="strategy for choosing max location in the attention map",
)
parser.add_argument(
    "--evaluation_method",
    type=str,
    default="inter_eye_distance",
    choices=["inter_eye_distance", "visible", "mean_average_error", "pck", "orientation_invariant"],
    help="strategy for evaluation",
)
parser.add_argument(
    "--min_dist",
    type=float,
    default=0.1,
    help="minimum distance between the keypoints, as a fraction of the image size",
)
parser.add_argument(
    "--furthest_point_num_samples",
    type=int,
    default=25,
    help="the number of samples to use if using the furthest point strategy",
)
parser.add_argument(
    "--num_indices",
    type=int,
    default=100,
    help="the number of samples to use for finding the indices of the best tokens",
)
parser.add_argument(
    "--num_subjects",
    type=int,
    default=1,
    help="the number of subjects within each image",
)
parser.add_argument(
    "--sharpening_loss_weight",
    type=float,
    default=100,
    help="Weight of the sharpening loss",
)
parser.add_argument(
    "--equivariance_attn_loss_weight",
    type=float,
    default=1000.0,
    help="Weight of the old equivariance loss",
)
parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument(
    "--noise_level",
    type=int,
    default=-1,
    help="noise level for the test set between 0 and 49 where 0 is the highest noise level and 49 is the lowest noise level",
)
parser.add_argument(
    "--max_num_points",
    type=int,
    default=50_000,
    help="number of samples to precompute",
)
parser.add_argument(
    "--sigma", type=float, default=2.0, help="sigma for the gaussian kernel"
)
parser.add_argument(
    "--augment_degrees",
    type=float,
    default=15.0,
    help="rotation degrees for augmentation",
)
parser.add_argument(
    "--augment_scale",
    type=float,
    # 2 arguments
    nargs="+",
    default=[0.8, 1.0],
    help="scale factor for augmentation",
)
parser.add_argument(
    "--augment_translate",
    type=float,
    nargs="+",
    default=[0.25, 0.25],
    help="amount of translation for augmentation along x and y axis",
)
parser.add_argument(
    "--augmentation_iterations",
    type=int,
    default=10,
    help="number of iterations for augmentation",
)
# store true the boolean argument 'visualize'
parser.add_argument(
    "--visualize", action="store_true", help="visualize the attention maps"
)
parser.add_argument(
    "--validation", action="store_true", help="use the validation sets instead of the training/testing set"
)
parser.add_argument("--top_k", type=int, default=10, help="number of points to choose")

args = parser.parse_args()

ldm, controllers, num_gpus = load_ldm(args.device, args.model_type, feature_upsample_res=args.feature_upsample_res, my_token=args.my_token)

# if args.save_folder doesnt exist create it
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
    
# print number of gpus
print("Number of GPUs: ", torch.cuda.device_count())

if args.wandb:
    # start a wandb session
    wandb.init(project="attention_maps", name=args.wandb_name, config=vars(args))


# Stage 1: Optimize Embedding (runs unconditionally)
embedding = optimize_embedding(
    ldm,
    args,
    controllers,
    num_gpus,
)
torch.save(embedding, os.path.join(args.save_folder, "embedding.pt"))
    
# Stage 2: Find Best Indices (runs unconditionally)
indices = find_best_indices(
    ldm,
    embedding,
    args,
    controllers,
    num_gpus,
)
torch.save(indices, os.path.join(args.save_folder, "indices.pt"))
    
if args.visualize:
    # Visualize embeddings after finding indices
    visualize_attn_maps(
        ldm,
        embedding,
        indices,
        args,
        controllers,
        num_gpus,
        # regressor is not available yet for the first visualization
    )

# Check for custom dataset before precomputation
if args.dataset_name == "custom":
    print("Dataset is 'custom'. Skipping precomputation, regressor training, and evaluation stages.")
    # If you want to exit completely after visualization for custom datasets:
    # import sys
    # sys.exit(0)
else:
    # Stage 3: Precompute Keypoints (runs if not custom dataset)
    source_kpts, target_kpts, visible = precompute_all_keypoints(
        ldm,
        embedding,
        indices,
        args,
        controllers,
        num_gpus,
    )

    torch.save(source_kpts, os.path.join(args.save_folder, "source_keypoints.pt"))
    torch.save(target_kpts, os.path.join(args.save_folder, "target_keypoints.pt"))
    if visible is not None: # visible can be None
        torch.save(visible, os.path.join(args.save_folder, "visible.pt"))

    # Stage 4: Train Regressor (runs if not custom dataset)
    if args.evaluation_method == "visible" or args.evaluation_method == "mean_average_error":
        if visible is None:
            # If visible is None from precompute (e.g. custom dataset didn't yield it, though we stop before this for custom)
            # or if a dataset type simply doesn't provide visibility.
            # Create a dummy visible tensor full of ones if it's required by evaluation but not provided.
            # This part might need adjustment based on how precompute_all_keypoints handles visible for all dataset types.
            # For now, assuming target_kpts is available to infer shape.
            visible_reshaped = torch.ones_like(target_kpts).reshape(target_kpts.shape[0], target_kpts.shape[1] * 2).cpu().numpy().astype(np.float64)
        else:
            visible_reshaped = visible.unsqueeze(-1).repeat(1, 1, 2).reshape(visible.shape[0], visible.shape[1] * 2).cpu().numpy().astype(np.float64)

        regressor = return_regressor_visible( 
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1]*2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1]*2).astype(np.float64),
            visible_reshaped,
        )
    elif args.evaluation_method == "orientation_invariant":
        regressor = return_regressor_human36m( 
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1]*2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1]*2).astype(np.float64),
        )
    else:
        regressor = return_regressor( 
            source_kpts.cpu().numpy().reshape(source_kpts.shape[0], source_kpts.shape[1]*2).astype(np.float64),
            target_kpts.cpu().numpy().reshape(target_kpts.shape[0], target_kpts.shape[1]*2).astype(np.float64),
        )
    regressor = torch.tensor(regressor).to(torch.float32)
    torch.save(regressor, os.path.join(args.save_folder, "regressor.pt"))

    if args.visualize:
        # Visualize with regressor (runs if not custom dataset and visualize is true)
        visualize_attn_maps(
            ldm,
            embedding,
            indices,
            args,
            controllers,
            num_gpus,
            regressor=regressor.to(args.device),
        )

    # Stage 5: Evaluate (runs if not custom dataset)
    evaluate(
        ldm,
        embedding,
        indices,
        regressor.to(args.device),
        args,
        controllers,
        num_gpus,
    )
