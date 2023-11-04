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

import torch.distributions as dist
import numpy as np
import torch
from typing import Optional, Union, Tuple, List, Dict
from tqdm.notebook import tqdm
import torch.nn.functional as F
import abc
from unsupervised_keypoints.eval import find_max_pixel
from unsupervised_keypoints import optimize_token
from torch.nn.parallel.data_parallel import DataParallel
from collections import OrderedDict

from PIL import Image

from unsupervised_keypoints.optimize import collect_maps


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, dict, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, dict, is_cross: bool, place_in_unet: str):
        
        dict = self.forward(dict, is_cross, place_in_unet)
        
        return dict['attn']

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
        return {
            "attn": [],
        }

    def forward(self, dict, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32**2:  # avoid memory overhead
        self.step_store["attn"].append(dict['attn']) 
        
        return dict

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()


def find_top_k_gaussian(attention_maps, top_k, min_dist=0.05, sigma = 3, epsilon = 1e-5):
    """
    attention_maps is of shape [batch_size, image_h, image_w]
    
    min_dist set to 0 becomes a simple top_k
    """
    
    
    
    device = attention_maps.device

    batch_size, image_h, image_w = attention_maps.shape

    max_pixel_locations = find_max_pixel(attention_maps)/image_h

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(attention_maps.view(batch_size, image_h * image_w)+epsilon, dim=-1)

    target = optimize_token.gaussian_circle(
        max_pixel_locations, size=image_h, sigma=sigma, device=attention_maps.device
    )  # Assuming H and W are the same
    
    target = target.reshape(batch_size, image_h * image_w)+epsilon
    target/=target.sum(dim=-1, keepdim=True)

    # sort the kl distances between attention_maps_softmax and target
    kl_distances = torch.sum(target * (torch.log(target) - torch.log(attention_maps_softmax)), dim=-1)
    # get the argsort for kl_distances
    kl_distances_argsort = torch.argsort(kl_distances, dim=-1, descending=False)
    
    
    selected_indices = [kl_distances_argsort[0]]
    
    this_index = 1
    
    while len(selected_indices) < top_k:
            
        # get the current index
        this_entropy_index = kl_distances_argsort[this_index]
        
        # get the location of the current index
        this_entropy_index_location = max_pixel_locations[this_entropy_index]
        
        # check if the location is far enough away from the other selected indices
        if torch.all(torch.sqrt(torch.sum((this_entropy_index_location - torch.index_select(max_pixel_locations, 0, torch.tensor(selected_indices).to(device)))**2, dim=-1)) > min_dist):
            selected_indices.append(this_entropy_index.item())
        
        this_index += 1
        
        # assert this_index < batch_size, "Not enough unique indices found"

    return torch.tensor(selected_indices).to(device)


def furthest_point_sampling(attention_maps, top_k, initial_candidates=30, sigma = 3, epsilon = 1e-5):
    """
    attention_maps is of shape [batch_size, image_h, image_w]
    
    min_dist set to 0 becomes a simple top_k
    """
    
    device = attention_maps.device

    batch_size, image_h, image_w = attention_maps.shape

    # Assuming you have a function find_max_pixel to get the pixel locations
    max_pixel_locations = find_max_pixel(attention_maps)/image_h  # You'll need to define find_max_pixel

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(attention_maps.view(batch_size, image_h * image_w) + epsilon, dim=-1)

    # Assuming you have a function gaussian_circle in the module optimize_token
    target = optimize_token.gaussian_circle(
        max_pixel_locations, size=image_h, sigma=sigma, device=attention_maps.device
    )  # You'll need to define gaussian_circle
    
    target = target.reshape(batch_size, image_h * image_w) + epsilon
    target /= target.sum(dim=-1, keepdim=True)

    # sort the kl distances between attention_maps_softmax and target
    kl_distances = torch.sum(target * (torch.log(target) - torch.log(attention_maps_softmax)), dim=-1)
    
    # get the argsort for kl_distances
    kl_distances_argsort = torch.argsort(kl_distances, dim=-1, descending=False)
    
    # Take top 30 points based on the kl divergence
    top_initial_candidates = kl_distances_argsort[:initial_candidates]
    
    if initial_candidates == top_k:
        return top_initial_candidates
    
    # Initialize the furthest point sampling
    selected_indices = [top_initial_candidates[0].item()]
    
    for _ in range(top_k - 1):
        max_min_dist = -1
        furthest_point = None
        
        for i in top_initial_candidates:
            if i.item() in selected_indices:
                continue
            
            this_min_dist = torch.min(torch.sqrt(torch.sum((max_pixel_locations[i] - torch.index_select(max_pixel_locations, 0, torch.tensor(selected_indices).to(device)))**2, dim=-1)))
            
            if this_min_dist > max_min_dist:
                max_min_dist = this_min_dist
                furthest_point = i.item()
        
        if furthest_point is not None:
            selected_indices.append(furthest_point)
    
    return torch.tensor(selected_indices).to(device)


def furthest_point_sampling_consistent(attention_maps, top_k, initial_candidates=30):
    """
    attention_maps is of shape [batch_size, image_h, image_w]
    
    min_dist set to 0 becomes a simple top_k
    """
    
    device = attention_maps.device

    batch_size, image_h, image_w = attention_maps.shape

    # Assuming you have a function find_max_pixel to get the pixel locations
    max_pixel_locations = find_max_pixel(attention_maps)/image_h  # You'll need to define find_max_pixel
    
    # Take top initial_candidates points consistently
    top_initial_candidates = torch.arange(top_k).to(device)
    
    if initial_candidates == top_k:
        return top_initial_candidates
    
    # Initialize the furthest point sampling
    selected_indices = [top_initial_candidates[0].item()]
    
    for _ in range(top_k - 1):
        max_min_dist = -1
        furthest_point = None
        
        for i in top_initial_candidates:
            if i.item() in selected_indices:
                continue
            
            this_min_dist = torch.min(torch.sqrt(torch.sum((max_pixel_locations[i] - torch.index_select(max_pixel_locations, 0, torch.tensor(selected_indices).to(device)))**2, dim=-1)))
            
            if this_min_dist > max_min_dist:
                max_min_dist = this_min_dist
                furthest_point = i.item()
        
        if furthest_point is not None:
            selected_indices.append(furthest_point)
    
    return torch.tensor(selected_indices).to(device)




def find_top_k(attention_maps, top_k, min_dist=0.05):
    """
    attention_maps is of shape [batch_size, image_h, image_w]
    
    min_dist set to 0 becomes a simple top_k
    """
    
    device = attention_maps.device
    
    batch_size, image_h, image_w = attention_maps.shape
    
    max_pixel_locations = find_max_pixel(attention_maps)/image_h

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(attention_maps.view(batch_size, image_h * image_w), dim=-1)

    # Compute the entropy of each token
    entropy = dist.Categorical(probs=attention_maps_softmax).entropy()
    
    # find the argsort for entropy
    entropy_argsort = torch.argsort(entropy, dim=-1, descending=False)
    
    selected_indices = [entropy_argsort[0]]
    
    this_index = 1
    
    while len(selected_indices) < top_k:
            
        # get the current index
        this_entropy_index = entropy_argsort[this_index]
        
        # get the location of the current index
        this_entropy_index_location = max_pixel_locations[this_entropy_index]
        
        # check if the location is far enough away from the other selected indices
        if torch.all(torch.sqrt(torch.sum((this_entropy_index_location - torch.index_select(max_pixel_locations, 0, torch.tensor(selected_indices).to(device)))**2, dim=-1)) > min_dist):
            selected_indices.append(this_entropy_index.item())
        
        this_index += 1
        
        # assert this_index < batch_size, "Not enough unique indices found"

    return torch.tensor(selected_indices).to(device)


def random_range(size, min_val, max_val, dtype=torch.float32):
    """
    Generate a random tensor of shape `size` with values in the range `[min_val, max_val]`.

    Parameters:
    - size (tuple): The shape of the output tensor.
    - min_val (float): The minimum value in the range.
    - max_val (float): The maximum value in the range.
    - dtype (torch.dtype, optional): The desired data type of the output tensor. Default is torch.float32.

    Returns:
    - torch.Tensor: A tensor of random numbers in the range `[min_val, max_val]`.
    """
    return torch.rand(size, dtype=dtype) * (max_val - min_val) + min_val

def find_pred_noise(
    ldm,
    image,
    context,
    noise_level=-1,
    device="cuda",
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()

    with torch.no_grad():
        latent = image2latent(ldm, image, device)
        
    noise = torch.randn_like(latent)

    noisy_image = ldm.scheduler.add_noise(
        latent, noise, ldm.scheduler.timesteps[noise_level]
    )
    
    # import ipdb; ipdb.set_trace()

    pred_noise = ldm.unet(noisy_image, 
                          ldm.scheduler.timesteps[noise_level].repeat(noisy_image.shape[0]), 
                          context.repeat(noisy_image.shape[0], 1, 1))["sample"]
    
    return noise, pred_noise
    

def run_and_find_attn(
    ldm,
    image,
    context,
    noise_level=-1,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    upsample_res=32,
    indices=None,
    controllers=None,
    human36m = False
):
    _, _ = find_pred_noise(
        ldm,
        image,
        context,
        noise_level=noise_level,
        device=device,
    )
    
    attention_maps=[]
    
    for controller in controllers:

        _attention_maps = collect_maps(
            controllers[controller],
            from_where=from_where,
            upsample_res=upsample_res,
            layers=layers,
            indices=indices,
        )
        
        if human36m:
            _attention_maps = mask_attn(image, _attention_maps)
        
        attention_maps.append(_attention_maps)

        controllers[controller].reset()
        
        

    return attention_maps


def mask_attn(image, attn_map):
    C, H, W = attn_map.shape
    # if  image is numpy array, convert to torch tensor
    if type(image) is np.ndarray:
        image = torch.from_numpy(image).permute(0, 3, 1, 2).to(attn_map.device)

    downsampled_img = F.interpolate(image, size=(H, W), mode="bilinear", align_corners=False)
    downsampled_img = downsampled_img.mean(dim=1).to(attn_map.device)
    # mask attn_maps where downsampled_img is 0
    attn_map = attn_map*(downsampled_img!=0)
        
    return attn_map


def image2latent(model, image, device):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            # print the max and min values of the image
            image = torch.from_numpy(image).float() * 2 - 1
            image = image.permute(0, 3, 1, 2).to(device)
            if device != "cpu":
                latents = model.vae.module.encode(image)["latent_dist"].mean
            else:
                latents = model.vae.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
    return latents


# class CustomDataParallel(DataParallel):
#     def gather(self, outputs, output_device):
#         # Assuming 'outputs' is a list of 'BaseOutput' from multiple GPUs
#         gathered_output = BaseOutput()
        
#         for output in outputs:
#             for k, v in output.items():
#                 if k not in gathered_output:
#                     gathered_output[k] = 0
#                 gathered_output[k] += v  # Example operation
                
#         return gathered_output
    
    

def diffusion_step(
    model, latents, context, t
):
    
    # import ipdb; ipdb.set_trace()  
    noise_pred = model.unet(latents, t.repeat(latents.shape[0]), context.repeat(latents.shape[0], 1, 1))["sample"]
    
    # import ipdb; ipdb.set_trace()  

    # latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    # latents = controller.step_callback(latents)
    return noise_pred


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(
        1, model.unet.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


# @torch.no_grad()
# def text2image_ldm(
#     model,
#     prompt: List[str],
#     controller,
#     num_inference_steps: int = 50,
#     guidance_scale: Optional[float] = 7.0,
#     generator: Optional[torch.Generator] = None,
#     latent: Optional[torch.FloatTensor] = None,
# ):
#     register_attention_control(model, controller)
#     height = width = 256
#     batch_size = len(prompt)

#     uncond_input = model.tokenizer(
#         [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
#     )
#     uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

#     text_input = model.tokenizer(
#         prompt, padding="max_length", max_length=77, return_tensors="pt"
#     )
#     text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
#     latent, latents = init_latent(latent, model, height, width, generator, batch_size)
#     context = torch.cat([uncond_embeddings, text_embeddings])

#     model.scheduler.set_timesteps(num_inference_steps)
#     for t in tqdm(model.scheduler.timesteps):
#         latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

#     image = latent2image(model.vqvae, latents)

#     return image, latent

# def latent_step(model, controller, latents, context, t):

#     noise_pred = model.unet(latents, t, encoder_hidden_states=context)["sample"]
#     latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
#     latents = controller.step_callback(latents)
#     return latents

def latent_step(model, controller, latents, context, t, guidance_scale, low_resource=True):
    if low_resource:
        # noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    # noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    noise_pred = noise_prediction_text
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents

def register_attention_control_generation(model, controller, target_attn_maps, indices):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            
            # if (
            #     is_cross
            #     and sequence_length <= 32**2
            #     and len(controller.step_store["attn"]) < 4
            #     and attn.shape[-1] != 77
            # ):
            #     attn = controller({"attn": attn}, is_cross, place_in_unet)
            #     target_res = int(attn.shape[1]**0.5)
                
            #     # downsample target_attn_maps to target_res
            #     downsampled = F.interpolate(
            #         target_attn_maps[None],
            #         size=(target_res, target_res),
            #         mode="bilinear",
            #         align_corners=False,
            #     )[0]
            #     downsampled = downsampled.reshape(-1, target_res**2)
            #     downsampled /= downsampled.max(dim=-1, keepdim=True)[0]
            #     downsampled = downsampled.permute(1, 0)
            #     downsampled = downsampled[None].repeat(attn.shape[0], 1, 1)
            
            #     attn[:, :, indices] += downsampled*2
                
            #     attn /= attn.sum(dim=-1, keepdim=True)
            
            
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward
    
    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    # replacing forward function in /scratch/iamerich/miniconda3/envs/LDM_correspondences/lib/python3.10/site-packages/diffusers/models/attention.py
    # line 518
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "CrossAttention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
            
    print("cross_att_count")
    print(cross_att_count)

    controller.num_att_layers = cross_att_count



@torch.no_grad()
def text2image_ldm_stable(
    model,
    embedding,
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    target_attn_maps = torch.load("example_attn_maps_indices.pt")
    indices = torch.load("outputs/indices.pt")
    register_attention_control_generation(model, controller, target_attn_maps, indices)
    height = width = 512
    
    
    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    # latent, latents = init_latent(latent, model, height, width, generator)
    
    latents = torch.randn(
        (1, model.unet.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(model.device)
    
    context = [uncond_embeddings, embedding]

    # set timesteps
    # extra_set_kwargs = {"offset": 1}
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = latent_step(model, controller, latents, context, t, guidance_scale=guidance_scale, low_resource = True)
        controller.reset()
    # latents = latent_step(model, controller, latents, context, t, guidance_scale=guidance_scale, low_resource = True)

    image = latent2image(model.vae, latents)

    return image, latent


def softmax_torch(x):  # Assuming x has atleast 2 dimensions
    maxes = torch.max(x, -1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs


def register_attention_control(model, controller, feature_upsample_res=256):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None

            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            # sim = torch.matmul(q, k.permute(0, 2, 1)) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim = sim.masked_fill(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = torch.nn.Softmax(dim=-1)(sim)
            attn = attn.clone()
            
            out = torch.matmul(attn, v)

            if (
                is_cross
                and sequence_length <= 32**2
                and len(controller.step_store["attn"]) < 4
            ):
                x_reshaped = x.reshape(
                    batch_size,
                    int(sequence_length**0.5),
                    int(sequence_length**0.5),
                    dim,
                ).permute(0, 3, 1, 2)
                # upsample to feature_upsample_res**2
                x_reshaped = (
                    F.interpolate(
                        x_reshaped,
                        size=(feature_upsample_res, feature_upsample_res),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(batch_size, -1, dim)
                )

                q = self.to_q(x_reshaped)
                q = self.reshape_heads_to_batch_dim(q)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = torch.nn.Softmax(dim=-1)(sim)
                attn = attn.clone()

                attn = controller({"attn": attn}, is_cross, place_in_unet)

            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    # replacing forward function in /scratch/iamerich/miniconda3/envs/LDM_correspondences/lib/python3.10/site-packages/diffusers/models/attention.py
    # line 518
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "CrossAttention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")

    controller.num_att_layers = cross_att_count


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=77,
):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i
        )
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [
                get_word_inds(prompts[i], key, tokenizer)
                for i in range(1, len(prompts))
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(
                        alpha_time_words, item, i, ind
                    )
    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )
    return alpha_time_words


def init_random_noise(device, num_words=77):
    return torch.randn(1, num_words, 768).to(device)


def find_latents(ldm, image, device="cuda"):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    with torch.no_grad():
        latent = image2latent(ldm, image, device)

    return latent
