import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from unsupervised_keypoints import ptp_utils

device = "cuda"

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


for i in range(100):

    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", use_auth_token=MY_TOKEN, scheduler=scheduler
    ).to(device)

    controller = ptp_utils.AttentionStore()

    embedding = torch.load("outputs/embedding.pt").to(device)

    image = ptp_utils.text2image_ldm_stable(
        model,
        embedding = embedding,
        controller = controller,
        num_inference_steps= 50,
    )

    import matplotlib.pyplot as plt
    plt.imshow(image[0][0])
    plt.savefig(f"outputs/image_{i:03d}.png")