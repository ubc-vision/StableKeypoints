import os
import argparse
import torch
from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.keypoint_regressor import LinearProjection
from unsupervised_keypoints.optimize import optimize_embedding

from unsupervised_keypoints.keypoint_regressor import (
    supervise_regressor,
    find_best_indices,
    precompute_all_keypoints,
    return_regressor,
)

from unsupervised_keypoints.eval import eval_embedding, evaluate
from unsupervised_keypoints.visualize import visualize_attn_maps


# Argument parsing
parser = argparse.ArgumentParser(description="optimize a class embedding")

# Network details
parser.add_argument(
    "--model_type",
    type=str,
    # default="CompVis/stable-diffusion-v1-4",
    default="runwayml/stable-diffusion-v1-5",
    help="ldm model type",
)
# Dataset details
parser.add_argument(
    "--celeba_loc",
    type=str,
    default="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/",
    help="Path to celeba dataset",
)
parser.add_argument(
    "--mafl_loc",
    type=str,
    default="/ubc/cs/home/i/iamerich/scratch/datasets/celeba/TCDCN-face-alignment/MAFL/",
    help="Path to mafl train test split",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="outputs",
    help="Where to save visualizations and checkpoints",
)
# make a term for sdxl, itll be bool and only true if we want to use sdxl
parser.add_argument("--sdxl", action="store_true", help="use sdxl")
parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
# boolean argument called wandb
parser.add_argument("--wandb", action="store_true", help="wandb logging")
# argument for learning rate
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
# add argument for num_steps
parser.add_argument(
    "--num_steps", type=int, default=1e4, help="number of steps to optimize"
)
parser.add_argument(
    "--num_tokens", type=int, default=1000, help="number of tokens to optimize"
)
parser.add_argument(
    "--max_num_images",
    type=int,
    default=19000,
    help="number of samples to use for finding regressor",
)
parser.add_argument("--layers", type=int, nargs="+", default=[5, 6, 7, 8])
parser.add_argument(
    "--noise_level",
    type=int,
    default=-8,
    help="noise level for the test set between 0 and 49 where 0 is the highest noise level and 49 is the lowest noise level",
)
parser.add_argument(
    "--kernel_size",
    type=int,
    default=3,
    help="size of blur over the ground truth attention map",
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
    default=[0.4, 1.0],
    help="scale factor for augmentation",
)
parser.add_argument(
    "--augment_translate",
    type=float,
    nargs="+",
    # default=[0.3, 0.3],
    default=[0.25, 0.25],
    help="amount of translation for augmentation along x and y axis",
)
parser.add_argument(
    "--augment_shear",
    type=float,
    nargs="+",
    default=[0.0, 0.0],
    help=" amount of shear for augmentation",
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
parser.add_argument("--top_k", type=int, default=10, help="number of points to choose")

args = parser.parse_args()

ldm = load_ldm(args.device, args.model_type)

# embedding = optimize_embedding(
#     ldm,
#     wandb_log=args.wandb,
#     lr=args.lr,
#     num_steps=int(args.num_steps),
#     num_tokens=args.num_tokens,
#     device=args.device,
#     layers=args.layers,
#     sdxl=args.sdxl,
#     top_k=args.top_k,
#     kernel_size=args.kernel_size,
#     augment_degrees=args.augment_degrees,
#     augment_scale=args.augment_scale,
#     augment_translate=args.augment_translate,
#     augment_shear=args.augment_shear,
#     mafl_loc=args.mafl_loc,
#     celeba_loc=args.celeba_loc,
# )
# torch.save(embedding, os.path.join(args.save_folder, "embedding.pt"))
embedding = torch.load("embedding.pt").to(args.device).detach()
#
# indices = find_best_indices(
#     ldm,
#     embedding,
#     num_steps=100,
#     num_tokens=args.num_tokens,
#     device=args.device,
#     layers=args.layers,
#     top_k=args.top_k,
#     augment=True,
#     augment_degrees=args.augment_degrees,
#     augment_scale=args.augment_scale,
#     augment_translate=args.augment_translate,
#     augment_shear=args.augment_shear,
#     mafl_loc=args.mafl_loc,
#     celeba_loc=args.celeba_loc,
# )
# torch.save(indices, os.path.join(args.save_folder, "indices.pt"))
indices = torch.load("indices.pt").to(args.device).detach()

# visualize embeddings
visualize_attn_maps(
    ldm,
    embedding,
    indices,
    num_tokens=args.num_tokens,
    layers=args.layers,
    num_points=args.top_k,
    augment_degrees=args.augment_degrees,
    augment_scale=args.augment_scale,
    augment_translate=args.augment_translate,
    augment_shear=args.augment_shear,
    augmentation_iterations=args.augmentation_iterations,
    mafl_loc=args.mafl_loc,
    celeba_loc=args.celeba_loc,
    save_folder=args.save_folder,
    visualize=args.visualize,
    device=args.device,
)

source_kpts, target_kpts = precompute_all_keypoints(
    ldm,
    embedding,
    indices,
    wandb_log=args.wandb,
    lr=1e-2,
    num_steps=1e4,
    num_tokens=args.num_tokens,
    device=args.device,
    layers=args.layers,
    top_k=args.top_k,
    augment_degrees=args.augment_degrees,
    augment_scale=args.augment_scale,
    augment_translate=args.augment_translate,
    augment_shear=args.augment_shear,
    augmentation_iterations=args.augmentation_iterations,
    max_num_images=args.max_num_images,
    mafl_loc=args.mafl_loc,
    celeba_loc=args.celeba_loc,
)


torch.save(source_kpts, os.path.join(args.save_folder, "source_keypoints.pt"))
torch.save(target_kpts, os.path.join(args.save_folder, "target_keypoints.pt"))
# source_kpts = torch.load("keypoints.pt")
# target_kpts = torch.load("target_keypoints.pt")

regressor = return_regressor(
    source_kpts.cpu().numpy().reshape(-1, 20),
    target_kpts.cpu().numpy().reshape(-1, 10),
)
regressor = torch.tensor(regressor)
torch.save(regressor, os.path.join(args.save_folder, "regressor.pt"))

# regressor = torch.load("regressor.pt")

# visualize embeddings
visualize_attn_maps(
    ldm,
    embedding,
    indices,
    num_tokens=args.num_tokens,
    layers=args.layers,
    num_points=args.top_k,
    regressor=regressor.to(args.device),
    augment_degrees=args.augment_degrees,
    augment_scale=args.augment_scale,
    augment_translate=args.augment_translate,
    augment_shear=args.augment_shear,
    mafl_loc=args.mafl_loc,
    celeba_loc=args.celeba_loc,
    save_folder=args.save_folder,
    device=args.device,
)

evaluate(
    ldm,
    embedding,
    indices,
    regressor.to(args.device),
    num_tokens=args.num_tokens,
    layers=args.layers,
    noise_level=args.noise_level,
    augment_degrees=args.augment_degrees,
    augment_scale=args.augment_scale,
    augment_translate=args.augment_translate,
    augment_shear=args.augment_shear,
    augmentation_iterations=args.augmentation_iterations,
    mafl_loc=args.mafl_loc,
    celeba_loc=args.celeba_loc,
    save_folder=args.save_folder,
    device=args.device,
)
