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
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
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
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
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
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def find_top_k(attention_maps, top_k):
    """
    attention_maps is of shape [batch_size, image_size]
    """

    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(attention_maps, dim=-1)

    # Compute the entropy of each token
    entropy = dist.Categorical(probs=attention_maps_softmax).entropy()

    # Select the top_k tokens with the lowest entropy
    _, top_embedding_indices = torch.topk(entropy, top_k, largest=False)

    return top_embedding_indices


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


def run_and_find_attn(
    ldm,
    image,
    context,
    noise_level=-1,
    device="cuda",
    from_where=["down_cross", "mid_cross", "up_cross"],
    layers=[0, 1, 2, 3, 4, 5],
    upsample_res=32,
):
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    with torch.no_grad():
        latent = image2latent(ldm, image, device)

    noisy_image = ldm.scheduler.add_noise(
        latent, torch.rand_like(latent), ldm.scheduler.timesteps[noise_level]
    )

    controller = AttentionStore()

    register_attention_control(ldm, controller)

    _ = diffusion_step(
        ldm,
        controller,
        noisy_image,
        context,
        ldm.scheduler.timesteps[noise_level],
        cfg=False,
    )

    attention_maps = collect_maps(
        controller,
        from_where=from_where,
        upsample_res=upsample_res,
        layers=layers,
    )

    # take the mean over the first 2 dimensions
    attention_maps = torch.mean(attention_maps, dim=(0, 1))

    # attention_maps[:, :, :3] = 0
    # attention_maps[:, :, -3:] = 0
    # attention_maps[:, :3, :] = 0
    # attention_maps[:, -3:, :] = 0

    return attention_maps


def image2latent(model, image, device):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            # print the max and min values of the image
            image = torch.from_numpy(image).float() * 2 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            latents = model.vae.encode(image)["latent_dist"].mean
            latents = latents * 0.18215
    return latents


def diffusion_step(
    model, controller, latents, context, t, guidance_scale=None, cfg=True
):
    if cfg:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
    else:
        noise_pred = model.unet(latents, t, encoder_hidden_states=context)["sample"]

    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents, noise_pred


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(
        batch_size, model.unet.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.0,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt, padding="max_length", max_length=77, return_tensors="pt"
    )
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vqvae, latents)

    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    # extra_set_kwargs = {"offset": 1}
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vae, latents)

    return image, latent


def softmax_torch(x):  # Assuming x has atleast 2 dimensions
    maxes = torch.max(x, -1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs


def register_attention_control(model, controller):
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
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.matmul(attn, v)

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
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

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