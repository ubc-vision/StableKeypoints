
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


def visualize_attention_map(attention_map, file_name):
    # save attention map
    attention_map = attention_map.unsqueeze(-1).repeat(1, 1, 3)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    attention_map = attention_map.detach().cpu().numpy()
    attention_map = (attention_map * 255).astype(np.uint8)
    img = Image.fromarray(attention_map)
    img.save(file_name)

def run_image_with_tokens(image_path, tokens, file_name):
    image = load_512(image_path)
    
    latent = image2latent(ldm_stable, image)
    
    controller = AttentionStore()
        
    ptp_utils.register_attention_control(ldm_stable, controller)
    
    latents = ptp_utils.diffusion_step(ldm_stable, controller, latent, tokens, torch.tensor(1), cfg=False)
    
    attention_maps = aggregate_attention(controller, 16, ["up", "down"], True, 0)
    
    # save image
    img = Image.fromarray(image)
    img.save(f"outputs/{file_name}_img.png")
    
    visualize_attention_map(attention_maps[..., 0], f"outputs/{file_name}_map.png")
    
    
def find_average_attention(image_path, ldm_stable, tokens, file_name):
    image = load_512(image_path)
    
    latent = image2latent(ldm_stable, image)
    
    controller = AttentionStore()
        
    ptp_utils.register_attention_control(ldm_stable, controller)
    
    latents = ptp_utils.diffusion_step(ldm_stable, controller, latent, tokens, torch.tensor(1), cfg=False)
    
    attention_maps = aggregate_attention(controller, 16, ["up", "down"], True, 0)
    
    # save image
    img = Image.fromarray(image)
    img.save(f"outputs/{file_name}_img.png")
    
    for i in range(attention_maps.shape[-1]):
        visualize_attention_map(attention_maps[..., i], f"outputs/{file_name}_map_{i}.png")
    
    visualize_attention_map(torch.mean(attention_maps, dim=-1), f"outputs/{file_name}_map.png")
    
    
def find_average_attention_from_list(image, ldm_stable, tokens, file_name=None, device='cuda', index=0):
    
    # if the image is a torch tensor, convert to numpy
    if type(image) is torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        
    latent = image2latent(ldm_stable, image, device=device)
    
    attention_maps = []
    
    for i in range(len(tokens)):
    
        controller = AttentionStore()
            
        ptp_utils.register_attention_control(ldm_stable, controller)
        
        latents = ptp_utils.diffusion_step(ldm_stable, controller, latent, tokens[i], torch.tensor(1), cfg=False)
        
        attention_map = aggregate_attention(controller, 16, ["up", "down"], True, 0)
        
        attention_maps.append(attention_map)
        
    attention_maps = torch.stack(attention_maps)
    
    attention_maps = torch.mean(attention_maps, dim=0)
    
    if file_name is not None:
        # make the image a uint8
        image = (image * 255).astype(np.uint8)
        # save image
        img = Image.fromarray(image)
        img.save(f"outputs/{file_name}_img.png")
        visualize_attention_map(attention_maps[..., index], f"outputs/{file_name}_map.png")
        
    return attention_maps[..., index]
        
    
    
    
def optimize_prompt(ldm, image, pixel_loc, context=None, device="cuda", num_steps=100):
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    with torch.no_grad():
        latent = image2latent(ldm, image, device)
        
    if context is None:
        context = init_random_noise(device)
        
    context.requires_grad = True
    
    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=0.001)
    
    # time the optimization
    import time
    start = time.time()
    
    for _ in range(num_steps):
        
        print("step")
        
        controller = AttentionStore()
        
        ptp_utils.register_attention_control(ldm, controller)
        
        _ = ptp_utils.diffusion_step(ldm, controller, latent, context, torch.tensor(1), cfg = False)
        
        attention_maps = aggregate_attention(controller, 16, ["up", "down"], True, 0)
        
            
        gt_maps = torch.zeros_like(attention_maps)
        
        x_loc = pixel_loc[0]*gt_maps.shape[0]
        y_loc = pixel_loc[1]*gt_maps.shape[1]
        
        # round x_loc and y_loc to the nearest integer
        x_loc = int(x_loc)
        y_loc = int(y_loc)
        
        gt_maps[int(y_loc), int(x_loc)] = 1
        
        
        # visualize_image_with_points(gt_maps[..., 0][None], [x_loc, y_loc], "gt_points")
        # visualize_image_with_points(image, [(x_loc+0.5)*512/16, (y_loc+0.5)*512/16], "gt_img_point_quantized")
        # visualize_image_with_points(image, [pixel_loc[0]*512, pixel_loc[1]*512], "gt_img_point")
        
        
        loss = torch.nn.MSELoss()(attention_maps[..., 0], gt_maps[..., 0])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # print the time it took to optimize
    print(f"optimization took {time.time() - start} seconds")
        

    return context

def optimize_prompt_informed(ldm, src_img, trg_img, pixel_loc, context=None, device="cuda", num_steps=100):
    
    # if src_img is a torch.tensor, convert to numpy
    if type(src_img) == torch.Tensor:
        src_img = src_img.permute(1, 2, 0).detach().cpu().numpy()
    if type(trg_img) == torch.Tensor:
        trg_img = trg_img.permute(1, 2, 0).detach().cpu().numpy()
    
    with torch.no_grad():
        latent_src = image2latent(ldm, src_img, device)
        latent_ref = image2latent(ldm, trg_img, device)
        
    if context is None:
        context = init_random_noise(device)
        
    context.requires_grad = True
    
    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=0.001)
    
    # time the optimization
    import time
    start = time.time()
    
    for _ in range(num_steps):
        
        controller = AttentionStore()
        
        ptp_utils.register_attention_control(ldm, controller)
        
        _ = ptp_utils.diffusion_step(ldm, controller, latent_src, context, torch.tensor(1), cfg = False)
        
        attention_maps = aggregate_attention(controller, 16, ["up", "down"], True, 0)
        
            
        gt_maps = torch.zeros_like(attention_maps)
        
        x_loc = pixel_loc[0]*gt_maps.shape[0]
        y_loc = pixel_loc[1]*gt_maps.shape[1]
        
        # round x_loc and y_loc to the nearest integer
        x_loc = int(x_loc)
        y_loc = int(y_loc)
        
        gt_maps[int(y_loc), int(x_loc)] = 1
        
        
        loss_src = torch.nn.MSELoss()(attention_maps[..., 0], gt_maps[..., 0])
        # loss = torch.nn.MSELoss()(attention_maps[..., 0], gt_maps[..., 0])
        # loss.backward()
        
        controller = AttentionStore()
        
        ptp_utils.register_attention_control(ldm, controller)
        
        _ = ptp_utils.diffusion_step(ldm, controller, latent_ref, context, torch.tensor(1), cfg = False)
        
        attention_maps = aggregate_attention(controller, 16, ["up", "down"], True, 0)
        
        # l1 loss on the attention maps
        loss_trg = (torch.norm(attention_maps[..., 0], p=1)-torch.max(torch.abs(attention_maps[..., 0])))*1e-5
        
        print("loss_trg, loss_src")
        print(loss_trg, loss_src)
        
        (loss_src+loss_trg).backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # print the time it took to optimize
    # print(f"optimization took {time.time() - start} seconds")
        

    return context

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


if __name__ == "__main__":

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm, _ = load_ldm(device)

    source_img = load_512("/scratch/iamerich/prompt-to-prompt/example_images/gnochi_mirror.jpeg")
    
    print("source_img.shape")
    print(source_img.shape)

    contexts = []
    for i in range(10):
        this_context = optimize_prompt(ldm, source_img, [0.9, 0.4])
        contexts.append(this_context.detach())
        
    target_img = load_512("/scratch/iamerich/prompt-to-prompt/example_images/cat1.jpeg")

    attn_map = find_average_attention_from_list(target_img, ldm, contexts, "avg_attn", index=0)
    max_val = find_max_pixel_value(attn_map)

    visualize_image_with_points(attn_map[None], max_val, "largest_loc")
    visualize_image_with_points(target_img, max_val*512/16, "largest_loc_img")


    exit()


    # find_corresponding_pixel(ldm, this_context, target_img)

    # find_average_attention("/scratch/iamerich/prompt-to-prompt/example_images/gnochi_mirror.jpeg", ldm_stable, context, "average_attention")
    # exit()

            
            
    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cat1.jpeg", context, "cat1_before")
    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cat2.jpeg", context, "cat2_before")
    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cat3.png", context, "cat3_before")
    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cats4.png", context, "cat4_before")

        



    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/gnochi_mirror.jpeg", context, "initial_test")

    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cat1.jpeg", context, "cat1_after")
    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cat2.jpeg", context, "cat2_after")
    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cat3.png", context, "cat3_after")
    run_image_with_tokens("/scratch/iamerich/prompt-to-prompt/example_images/cats4.png", context, "cat4_after")
