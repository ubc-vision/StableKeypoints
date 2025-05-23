# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
from unsupervised_keypoints import ptp_utils
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn


def load_ldm(device, type="CompVis/stable-diffusion-v1-4", feature_upsample_res=256, my_token=None):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    NUM_DDIM_STEPS = 50
    scheduler.set_timesteps(NUM_DDIM_STEPS)


    ldm = StableDiffusionPipeline.from_pretrained(
        type, use_auth_token=my_token, scheduler=scheduler
    ).to(device)
    
    if device != "cpu":
        ldm.unet = nn.DataParallel(ldm.unet)
        ldm.vae = nn.DataParallel(ldm.vae)
        
        controllers = {}
        for device_id in ldm.unet.device_ids:
            _device = torch.device("cuda", device_id)
            controller = ptp_utils.AttentionStore()
            controllers[_device] = controller
        effective_num_gpus = len(ldm.unet.device_ids)
    else:
        controllers = {}
        _device = torch.device("cpu")
        controller = ptp_utils.AttentionStore()
        controllers[_device] = controller
        effective_num_gpus = 1

        # patched_devices = set()

    def hook_fn(module, input):
        _device = input[0].device
        # if device not in patched_devices:
        ptp_utils.register_attention_control(module, controllers[_device], feature_upsample_res=feature_upsample_res)
        # patched_devices.add(device)

    if device != "cpu":
        ldm.unet.module.register_forward_pre_hook(hook_fn)
    else:
        ldm.unet.register_forward_pre_hook(hook_fn)
    
    for param in ldm.vae.parameters():
        param.requires_grad = False
    for param in ldm.text_encoder.parameters():
        param.requires_grad = False
    for param in ldm.unet.parameters():
        param.requires_grad = False

    return ldm, controllers, effective_num_gpus


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def init_prompt(model, prompt: str):
    uncond_input = model.tokenizer(
        [""],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    prompt = prompt

    return context, prompt


def latent2image(model, latents):
    latents = 1 / 0.18215 * latents
    image = model.vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def reshape_attention(attention_map):
    """takes average over 0th dimension and reshapes into square image

    Args:
        attention_map (4, img_size, -1): _description_
    """
    attention_map = attention_map.mean(0)
    img_size = int(np.sqrt(attention_map.shape[0]))
    attention_map = attention_map.reshape(img_size, img_size, -1)
    return attention_map


def visualize_attention_map(attention_map, file_name):
    # save attention map
    attention_map = attention_map.unsqueeze(-1).repeat(1, 1, 3)
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min()
    )
    attention_map = attention_map.detach().cpu().numpy()
    attention_map = (attention_map * 255).astype(np.uint8)
    img = Image.fromarray(attention_map)
    img.save(file_name)


def upscale_to_img_size(
    controller,
    from_where=["down_cross", "mid_cross", "up_cross"],
    upsample_res=512,
    layers=[0, 1, 2, 3, 4, 5],
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
            )[None, :, :, :, 1]

            if upsample_res != -1:
                # bilinearly upsample the image to img_sizeximg_size
                img = F.interpolate(
                    img,
                    size=(upsample_res, upsample_res),
                    mode="bilinear",
                    align_corners=False,
                )

            imgs.append(img)

    imgs = torch.cat(imgs, dim=0)

    return imgs


def gaussian_circle(pos, size=64, sigma=16, device="cuda"):
    """Create a batch of 2D Gaussian circles with a given size, standard deviation, and center coordinates.

    pos is in between 0 and 1 and has shape [batch_size, 2]

    """
    batch_size = pos.shape[0]
    _pos = pos * size  # Shape [batch_size, 2]
    _pos = _pos.unsqueeze(1).unsqueeze(1)  # Shape [batch_size, 1, 1, 2]

    grid = torch.meshgrid(torch.arange(size).to(device), torch.arange(size).to(device))
    grid = torch.stack(grid, dim=-1) + 0.5  # Shape [size, size, 2]
    grid = grid.unsqueeze(0)  # Shape [1, size, size, 2]

    dist_sq = (grid[..., 1] - _pos[..., 1]) ** 2 + (
        grid[..., 0] - _pos[..., 0]
    ) ** 2  # Shape [batch_size, size, size]
    dist_sq = -1 * dist_sq / (2.0 * sigma**2.0)
    gaussian = torch.exp(dist_sq)  # Shape [batch_size, size, size]

    return gaussian

def gaussian_circles(pos, size=64, sigma=16, device="cuda"):
    """In the case of multiple points, pos has shape [batch_size, num_points, 2]
    """
    
    circles = []

    for i in range(pos.shape[0]):
        _circles = gaussian_circle(
            pos[i], size=size, sigma=sigma, device=device
        )  # Assuming H and W are the same
        
        circles.append(_circles)
        
    circles = torch.stack(circles)
    circles = torch.mean(circles, dim=0)
    
    return circles