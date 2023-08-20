import argparse
import torch
from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.keypoint_regressor import LinearProjection
from unsupervised_keypoints.optimize import optimize_embedding, optimize_embedding_ddpm

from unsupervised_keypoints.keypoint_regressor import (
    supervise_regressor,
    find_best_indices,
)

from unsupervised_keypoints.eval import eval_embedding, evaluate
from unsupervised_keypoints.visualize import visualize_attn_maps


# Argument parsing
parser = argparse.ArgumentParser(description="optimize a class embedding")

# Network details
parser.add_argument(
    "--model_type",
    type=str,
    default="CompVis/stable-diffusion-v1-4",
    help="ldm model type",
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
parser.add_argument("--layers", type=int, nargs="+", default=[5, 6, 7, 8])
parser.add_argument(
    "--noise_level",
    type=int,
    default=-8,
    help="noise level for the test set between 0 and 49 where 0 is the highest noise level and 49 is the lowest noise level",
)
parser.add_argument(
    "--crop_percent",
    type=float,
    default=93.16549294381423,
    help="the percent of the image to crop to",
)
parser.add_argument("--top_k", type=int, default=30, help="number of points to choose")

args = parser.parse_args()

ldm = load_ldm(args.device, args.model_type)

embedding = optimize_embedding(
    ldm,
    wandb_log=args.wandb,
    lr=args.lr,
    num_steps=int(args.num_steps),
    num_tokens=args.num_tokens,
    device=args.device,
    layers=args.layers,
    sdxl=args.sdxl,
    top_k=args.top_k,
)
torch.save(embedding, "embedding.pt")
# embedding = torch.load("embedding.pt").to(args.device).detach()

indices = find_best_indices(
    ldm,
    embedding,
    num_steps=100,
    num_tokens=args.num_tokens,
    device=args.device,
    layers=args.layers,
    top_k=args.top_k,
    augment=False,
)
torch.save(indices, "indices.pt")
# indices = torch.load("indices.pt").to(args.device).detach()

# visualize embeddings
visualize_attn_maps(
    ldm,
    embedding,
    indices,
    num_tokens=args.num_tokens,
    layers=args.layers,
    num_points=args.top_k,
    augment=True,
)

regressor = supervise_regressor(
    ldm,
    embedding,
    indices,
    wandb_log=args.wandb,
    anneal_after_num_steps=1e3,
    lr=1e-3,
    num_steps=1e3,
    num_tokens=args.num_tokens,
    device=args.device,
    layers=args.layers,
    top_k=args.top_k,
)
torch.save(regressor.state_dict(), "regressor.pt")
# regressor = LinearProjection(input_dim=args.top_k * 2, output_dim=5 * 2).cuda()
# regressor_weights = torch.load("regressor.pt")
# regressor.load_state_dict(regressor_weights)

# visualize embeddings
visualize_attn_maps(
    ldm,
    embedding,
    indices,
    num_tokens=args.num_tokens,
    layers=args.layers,
    num_points=args.top_k,
    regressor=regressor,
    augment=False,
)

evaluate(
    ldm,
    embedding,
    indices,
    regressor,
    num_tokens=args.num_tokens,
    layers=args.layers,
    noise_level=args.noise_level,
    crop_percent=args.crop_percent,
)
