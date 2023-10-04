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
import abc
from unsupervised_keypoints import ptp_utils
from PIL import Image

import time
import torch.nn.functional as F

import torch.nn as nn

import pynvml


def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024**2


def load_ldm(device, type="CompVis/stable-diffusion-v1-4"):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    MY_TOKEN = ""
    NUM_DDIM_STEPS = 50
    scheduler.set_timesteps(NUM_DDIM_STEPS)

    ldm = StableDiffusionPipeline.from_pretrained(
        type, use_auth_token=MY_TOKEN, scheduler=scheduler
    ).to(device)
    
    controller = ptp_utils.AttentionStore()

    ptp_utils.register_attention_control(ldm, controller)
    

    for param in ldm.vae.parameters():
        param.requires_grad = False
    for param in ldm.text_encoder.parameters():
        param.requires_grad = False
    for param in ldm.unet.parameters():
        param.requires_grad = False

    return ldm, controller


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


def softargmax2d(input, beta=1000):
    *_, h, w = input.shape

    assert h == w, "only square images are supported"

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(input * beta, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w), np.linspace(0, 1, h), indexing="xy"
    )

    indices_r = (
        torch.tensor(np.reshape(indices_r, (-1, h * w))).to(input.device).float()
    )
    indices_c = (
        torch.tensor(np.reshape(indices_c, (-1, h * w))).to(input.device).float()
    )

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_c, result_r], dim=-1)

    return result / h


def find_context(image, ldm, pixel_loc, context_estimator, device="cuda"):
    with torch.no_grad():
        latent = image2latent(ldm, image.numpy().transpose(1, 2, 0), device)

    context = context_estimator(latent, pixel_loc)

    return context


def visualize_image_with_points(image, point, name, save_folder="outputs"):
    """The point is in pixel numbers"""

    import matplotlib.pyplot as plt

    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        try:
            image = image.permute(1, 2, 0).detach().cpu().numpy()
        except:
            import ipdb

            ipdb.set_trace()

    # make the figure without a border
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10, 10)

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(image, aspect="auto")

    if point is not None:
        # plot point on image
        plt.scatter(point[0].cpu(), point[1].cpu(), s=20, marker="o", c="r")

    plt.savefig(f"{save_folder}/{name}.png", dpi=200)
    plt.close()


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


def crop_image(image, pixel, crop_percent=80, margin=0.15):
    """pixel is an integer between 0 and image.shape[1] or image.shape[2]"""

    assert 0 < crop_percent <= 100, "crop_percent should be between 0 and 100"

    height, width, channels = image.shape
    crop_height = int(height * crop_percent / 100)
    crop_width = int(width * crop_percent / 100)

    # Calculate the crop region's top-left corner
    x, y = pixel

    # Calculate safe margin
    safe_margin_x = int(crop_width * margin)
    safe_margin_y = int(crop_height * margin)

    x_start_min = max(0, x - crop_width + safe_margin_x)
    x_start_min = min(x_start_min, width - crop_width)
    x_start_max = max(0, x - safe_margin_x)
    x_start_max = min(x_start_max, width - crop_width)

    y_start_min = max(0, y - crop_height + safe_margin_y)
    y_start_min = min(y_start_min, height - crop_height)
    y_start_max = max(0, y - safe_margin_y)
    y_start_max = min(y_start_max, height - crop_height)

    # Choose a random top-left corner within the allowed bounds
    x_start = torch.randint(int(x_start_min), int(x_start_max) + 1, (1,)).item()
    y_start = torch.randint(int(y_start_min), int(y_start_max) + 1, (1,)).item()

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

    # calculate new pixel location
    new_pixel = torch.stack([x - x_start, y - y_start])
    new_pixel = new_pixel / crop_width

    return (
        cropped_image.permute(1, 2, 0).numpy(),
        new_pixel,
        y_start,
        crop_height,
        x_start,
        crop_width,
    )
