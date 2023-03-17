
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

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image

import torch.nn.functional as F


# import ipdb

# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from time import sleep

import pynvml


def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2


def load_ldm(device):

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = ''
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
    # ldm_stable = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(device)

    
    
    # model_id = "stabilityai/stable-diffusion-2-1-base"

    # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    # ldm = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    # ldm = ldm.to(device)
    
    try:
        ldm.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm.tokenizer

    # ldm.scheduler.set_timesteps(NUM_DDIM_STEPS)


    for param in ldm.vae.parameters():
        param.requires_grad = False
    for param in ldm.text_encoder.parameters():
        param.requires_grad = False
    for param in ldm.unet.parameters():
        param.requires_grad = False
        
    return ldm, tokenizer
        

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):

        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return  0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):

        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):

        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):

        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        



def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    
    out = []
    attention_maps = attention_store.get_average_attention()
    
    import ipdb; ipdb.set_trace()
    
    # for key in attention_maps:
    #     print(key, attention_maps[key].shape)
    # print("attention_maps")
    # print(attention_maps)

    
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()



def extract_attention_map(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    
    # for key in attention_maps:
    #     print(key, attention_maps[key].shape)
    # print("attention_maps")
    # print(attention_maps)

    
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()



def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, prompts: List[str] = None, epoch: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.detach().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
        
    # ptp_utils.view_images(np.stack(images, axis=0))
    # save images
    for i, image in enumerate(images):
        Image.fromarray(image).save(f"outputs/cross_attention_{i}_{epoch:03d}.png")
    
    
    
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def init_prompt(model, prompt: str):
    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=model.tokenizer.model_max_length,
        return_tensors="pt"
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

def init_random_noise(device):
    return torch.randn(1, 77, 768).to(device)

def image2latent(model, image, device):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            latents = model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents


def run_image(model, image_path, prompt):
    image = load_512(image_path)

    latent = image2latent(model, image)
    # image_rec = null_inversion.latent2image(latent)

    ptp_utils.view_images(image)
    # ptp_utils.view_images(image_rec)

    # take the latents and pass them through a single diffusion step
    controller = AttentionStore()

    ptp_utils.register_attention_control(ldm_stable, controller)

    context, prompt = init_prompt(model, prompt)

    prompts = [prompt]

    with torch.no_grad():
        latents = ptp_utils.diffusion_step(ldm_stable, controller, latent, context, ldm_stable.scheduler.timesteps[-1], guidance_scale=GUIDANCE_SCALE)

    show_cross_attention(controller, 16, ["up", "down"])


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
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    attention_map = attention_map.detach().cpu().numpy()
    attention_map = (attention_map * 255).astype(np.uint8)
    img = Image.fromarray(attention_map)
    img.save(file_name)
    

def run_image_with_tokens(ldm, image, tokens, device='cuda', from_where = ["up_cross"], index=0, img_size=512):
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    latent = image2latent(ldm, image, device=device)
    
    controller = AttentionStore()
        
    ptp_utils.register_attention_control(ldm, controller)
    
    latents = ptp_utils.diffusion_step(ldm, controller, latent, tokens, torch.tensor(1), cfg=False)
    
    attention_maps = upscale_to_img_size(controller, from_where = from_where, img_size=img_size)
    # attention_maps = aggregate_attention(controller, map_size, from_where, True, 0)
    
    print("attention_maps.shape")
    print(attention_maps.shape)
    
    return attention_maps
    
    


def upscale_to_img_size(controller, from_where = ["up_cross"], img_size=512):
    """
    from_where is one of "down_cross" "mid_cross" "up_cross"
    
    returns the bilinearly upsampled attention map of size img_size x img_size for the first word in the prompt
    """
    
    imgs = []
    
    for key in from_where:
        for layer in range(len(controller.attention_store[key])):
            
            img = controller.attention_store[key][layer]
            
            img = img.reshape(4, int(img.shape[1]**0.5), int(img.shape[1]**0.5), img.shape[2])[None, :, :, :, 0]
            
            # import ipdb; ipdb.set_trace()
            # bilinearly upsample the image to img_sizeximg_size
            img = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)

            imgs.append(img)
            
    imgs = torch.cat(imgs, dim=0)
    
    return imgs
        
    
    
    
def optimize_prompt(ldm, image, pixel_loc, context=None, device="cuda", num_steps=100, from_where = ["up_cross"], img_size = 64):
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    with torch.no_grad():
        latent = image2latent(ldm, image, device)
        
    if context is None:
        context = init_random_noise(device)
        
    context.requires_grad = True
    
    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=1e-2)
    
    # time the optimization
    import time
    start = time.time()
    
    for _ in range(num_steps):
        
        controller = AttentionStore()
        
        ptp_utils.register_attention_control(ldm, controller)
        
        _ = ptp_utils.diffusion_step(ldm, controller, latent, context, torch.tensor(1), cfg = False)
        
        # attention_maps = aggregate_attention(controller, map_size, from_where, True, 0)
        attention_maps = upscale_to_img_size(controller, from_where = from_where, img_size=img_size)
        num_maps = attention_maps.shape[0]
        
        
        
        # divide by the mean along the dim=1
        attention_maps = torch.mean(attention_maps, dim=1)

        
            
        gt_maps = torch.zeros_like(attention_maps)
        

        
        x_loc = pixel_loc[0]*img_size
        y_loc = pixel_loc[1]*img_size
        
        # round x_loc and y_loc to the nearest integer
        x_loc = int(x_loc)
        y_loc = int(y_loc)
        
        gt_maps[:, int(y_loc), int(x_loc)] = 1
        
        gt_maps = gt_maps.reshape(num_maps, -1)
        attention_maps = attention_maps.reshape(num_maps, -1)
        
        print("attention_maps[0]")
        print(attention_maps[0])
        
        attention_maps = torch.softmax(attention_maps, dim=1)
        
        print("attention_maps[0]")
        print(attention_maps[0])
        
        
        # visualize_image_with_points(gt_maps[0], [0, 0], "gt_points")
        # visualize_image_with_points(image, [(x_loc+0.5), (y_loc+0.5)], "gt_img_point_quantized")
        # visualize_image_with_points(image, [pixel_loc[0]*512, pixel_loc[1]*512], "gt_img_point")
        # exit()
        
        
        # loss = torch.nn.MSELoss()(attention_maps[..., 0], gt_maps[..., 0])
        loss = torch.nn.CrossEntropyLoss()(attention_maps, gt_maps)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        print(loss.item())
        
    # print the time it took to optimize
    print(f"optimization took {time.time() - start} seconds")
        

    return context


@torch.no_grad()
def visualize_keypoints_over_subject(ldm, img, contexts, name, device):
    
    
    # if src_img is a torch.tensor, convert to numpy
    if type(img) == torch.Tensor:
        img = img.permute(1, 2, 0).detach().cpu().numpy()
    
    latent = image2latent(ldm, img, device)
    
    attention_maps = []
    
    
    for context in contexts:
    
        controller = AttentionStore()
            
        ptp_utils.register_attention_control(ldm, controller)
        
        _ = ptp_utils.diffusion_step(ldm, controller, latent, context, torch.tensor(1), cfg = False)
        
        attention_map = aggregate_attention(controller, 16, ["up", "down"], True, 0)
        
        attention_maps.append(attention_map[..., 0])
        
    attention_maps = torch.stack(attention_maps, dim=0)
        
    attention_maps_mean = torch.mean(attention_maps, dim=0, keepdim=True)
    
    attention_maps -= attention_maps_mean
    
    for i in range(attention_maps.shape[0]):
        attention_map = attention_maps[i]
        max_pixel = find_max_pixel_value(attention_map)
        visualize_image_with_points(attention_map[None], max_pixel, f'{name}_{i}')
        visualize_image_with_points(img, (max_pixel+0.5)*512/16, f'{name}_{i}_img')
        
    return attention_maps
        
        
    
    

def find_max_pixel_value(tens):
    """finds the 2d pixel location that is the max value in the tensor

    Args:
        tens (tensor): shape (height, width)
    """
    
    height = tens.shape[0]
    
    tens = tens.reshape(-1)
    max_loc = torch.argmax(tens)
    max_pixel = torch.stack([max_loc % height, torch.div(max_loc, height, rounding_mode='floor')])
    
    return max_pixel

def visualize_image_with_points(image, point, name):
    import matplotlib.pyplot as plt
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    plt.imshow(image)
    
    # plot point on image
    plt.scatter(point[0], point[1], s=3, marker='o', c='r')
    
    
    plt.savefig(f'outputs/{name}.png')
    plt.close()

