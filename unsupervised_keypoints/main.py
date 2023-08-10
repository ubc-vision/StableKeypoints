import argparse
import torch
from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.optimize import optimize_embedding, optimize_embedding_ddpm

from unsupervised_keypoints.eval import eval_embedding
from unsupervised_keypoints.visualize import visualize


# Argument parsing
parser = argparse.ArgumentParser(description='optimize a class embedding')

# Network details
parser.add_argument('--model_type', type=str,
                    default='CompVis/stable-diffusion-v1-4', help='ldm model type')
# make a term for sdxl, itll be bool and only true if we want to use sdxl
parser.add_argument('--sdxl', action='store_true', help='use sdxl')
parser.add_argument('--device', type=str,
                    default='cuda:0', help='device to use')
# boolean argument called wandb
parser.add_argument('--wandb', action='store_true', help='wandb logging')
# argument for learning rate
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
# add argument for num_steps
parser.add_argument('--num_steps', type=int, default=1e4, help='number of steps to optimize')
parser.add_argument('--num_tokens', type=int, default=1000, help='number of tokens to optimize')
parser.add_argument('--layers', type=int, nargs='+', default=[5, 6, 7, 8])
parser.add_argument('--top_k', type=int, default=10, help='number of points to choose')

args = parser.parse_args()

ldm = load_ldm(args.device, args.model_type)

embedding = optimize_embedding(ldm, wandb_log=args.wandb, lr = args.lr, num_steps=int(args.num_steps), num_tokens = args.num_tokens, device=args.device, layers=args.layers)
# embedding = optimize_embedding_ddpm(ldm, wandb_log=args.wandb, lr = args.lr, num_steps=int(args.num_steps), num_tokens = args.num_tokens, device=args.device, top_k=args.top_k)

# save embedding
torch.save(embedding, "embedding.pt")
# exit()

# embedding = torch.load("embedding_longest_run.pt").to(args.device).detach()
# embedding = torch.load("embedding.pt").to(args.device).detach()

# eval_embedding(ldm, embedding)
visualize(ldm, embedding, num_tokens= args.num_tokens, layers=args.layers)

