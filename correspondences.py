
import abc
import torch
import ptp_utils
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import autocast
from torch.optim.adam import Adam
import torch.nn.functional as nnf
from diffusers import StableDiffusionPipeline, DDIMScheduler

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

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
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
        
def null_optimization(self, latents, num_inner_steps, epsilon):
    uncond_embeddings, cond_embeddings = self.context.chunk(2)
    uncond_embeddings_list = []
    latent_cur = latents[-1]
    bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
    for i in range(NUM_DDIM_STEPS):
        uncond_embeddings = uncond_embeddings.clone().detach()
        uncond_embeddings.requires_grad = True
        optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
        latent_prev = latents[len(latents) - i - 2]
        t = self.model.scheduler.timesteps[i]
        with torch.no_grad():
            noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
        for j in range(num_inner_steps):
            noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
            latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
            loss = nnf.mse_loss(latents_prev_rec, latent_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            bar.update()
            if loss_item < epsilon + i * 2e-5:
                break
        for j in range(j + 1, num_inner_steps):
            bar.update()
        uncond_embeddings_list.append(uncond_embeddings[:1].detach())
        with torch.no_grad():
            context = torch.cat([uncond_embeddings, cond_embeddings])
            latent_cur = self.get_noise_pred(latent_cur, t, False, context)
    bar.close()
    return uncond_embeddings_list



NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
LOW_RESOURCE = False 
device = "cuda" if torch.cuda.is_available() else "cpu"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
model.disable_xformers_memory_efficient_attention()

# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# # num_samples = 10

# # prompt = num_samples*["a person swimming"]

# # image = pipe(prompt).images
# # print(len(image))
# # for i in range(len(image)):
# #     image[i].save(f"outputs/output_{i:02d}.png")
# # exit()

# controller = AttentionStore()

# image_path = "/scratch/iamerich/prompt-to-prompt/example_images/378-3780801_david-murray-news-person-full-body-png.jpeg"




image_gt = load_512(image_path)

prompt = "hand"

batch_size = len(prompt)
ptp_utils.register_attention_control(model, controller)
height = width = 512


text_input = model.tokenizer(
    prompt,
    padding="max_length",
    max_length=model.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)