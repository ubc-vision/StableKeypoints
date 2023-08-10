# load the dataset
import torch
import numpy as np
from tqdm import tqdm
from unsupervised_keypoints import ptp_utils
import torch.nn.functional as F
from unsupervised_keypoints.cats_dogs import Cats_Dogs

# now import weights and biases
import wandb


from unsupervised_keypoints.optimize_token import init_random_noise, image2latent, AttentionStore

from unsupervised_keypoints.optimize import collect_maps

def get_attn_map(ldm, image, context, device = 'cuda', from_where = ["down_cross", "mid_cross", "up_cross"], upsample_res = 32, layers = [0, 1, 2, 3, 4, 5], noise_level = -1, num_tokens = 77 ):
    
    # if image is a torch.tensor, convert to numpy
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).detach().cpu().numpy()
    
    with torch.no_grad():
        
        latent = image2latent(ldm, image, device)
        
    noisy_image = ldm.scheduler.add_noise(latent, torch.rand_like(latent), ldm.scheduler.timesteps[noise_level])
    
    controller = AttentionStore()
    
    ptp_utils.register_attention_control(ldm, controller)
    
    _ = ptp_utils.diffusion_step(ldm, controller, noisy_image, context, ldm.scheduler.timesteps[noise_level], cfg = False)
    
    attention_maps = collect_maps(controller, from_where = from_where, upsample_res=upsample_res, layers = layers, number_of_maps = num_tokens)
    
    attention_maps = torch.mean(attention_maps, dim = (0, 1))
    
    return attention_maps


def run_example(ldm, batch, context, threshold_value = 0.5, device = 'cuda', from_where = ["down_cross", "mid_cross", "up_cross"], upsample_res = 32, layers = [0, 1, 2, 3, 4, 5], noise_level = -1):
    
    image = batch['img']
    label = batch['label']
    
    attention_maps = get_attn_map(ldm, image, context, device = device, from_where = from_where, upsample_res = upsample_res, layers = layers, noise_level = noise_level)
    num_maps = attention_maps.shape[0]
    
    # the label should just be the maximum of the attention maps
    attn_map_max = torch.max(attention_maps)
    
    
    # compare threshold_value to attn_map_max with label
    # if below threshold_value, then label should be 0, if above threshold_value, then label should be 1
    est_label = (attn_map_max > threshold_value).to(torch.float)
    
    bool_correct = (est_label == label).to(torch.float)
    
    return bool_correct, attn_map_max

def eval_embedding(ldm, context, device="cuda", num_steps=2000, from_where = ["down_cross", "mid_cross", "up_cross"], upsample_res = 32, layers = [0, 1, 2, 3, 4, 5], lr=5e-3, noise_level = -1):
    
    
    dogs = Cats_Dogs(dogs=True, train=False)
    cats = Cats_Dogs(dogs=False, train=False)
    
    if context is None:
        context = init_random_noise(device)
        
    context.requires_grad = True
    
    # optimize context to maximize attention at pixel_loc
    optimizer = torch.optim.Adam([context], lr=lr)
    
    dog_accuracies = []
    est_dog_maxes = []
    
    for i in tqdm(range(len(dogs))):
        batch = dogs[i]
        accuracy, est_max = run_example(ldm, batch, context, threshold_value=0.1)
        dog_accuracies.append(accuracy.cpu().numpy())
        est_dog_maxes.append(est_max.detach().cpu().numpy())
    
    cat_accuracies = []
    est_cat_maxes = []
    for i in tqdm(range(len(cats))):
        batch = cats[i]
        accuracy, est_max = run_example(ldm, batch, context, threshold_value=0.1)
        cat_accuracies.append(accuracy.cpu().numpy())
        est_cat_maxes.append(est_max.detach().cpu().numpy())
        
    print(f"dog accuracy: {np.mean(dog_accuracies)}")
    print(f"cat accuracy: {np.mean(cat_accuracies)}")
    print(f"overall accuracy: {np.mean(dog_accuracies + cat_accuracies)}")
    
    # save the bins for the dog and cat maxes with matplotlib
    import matplotlib.pyplot as plt
    plt.hist(est_dog_maxes, bins=100, alpha=0.5, label='dog')
    plt.hist(est_cat_maxes, bins=100, alpha=0.5, label='cat')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("dog_cat_maxes.png")
    
        
    return np.mean(dog_accuracies + cat_accuracies)


def find_max_pixel(map):
    """
    finds the pixel of the map with the highest value
    map shape [batch_size, h, w]
    """
    
    batch_size, h, w = map.shape
    
    map = map.view(batch_size, -1)
    
    max_indices = torch.argmax(map, dim=-1)
    
    max_indices = max_indices.view(batch_size, 1)
    
    max_indices = torch.cat([max_indices // w, max_indices % w], dim=-1)
    
    return max_indices
    



def find_corresponding_points(maps, num_points=10):
    """
    Finds the corresponding points between the maps.
    Selects a set of maps with the common lowest entropy 
    Argmax of each of these is the corresponding point
    
    map shape [num_images, num_tokens, h, w]
    
    returns maximum pixel location and indices of the maps with the lowest entropy
    """
    
    num_images, num_tokens, h, w = maps.shape
    
    import torch.distributions as dist
        
    # Normalize the activation maps to represent probability distributions
    attention_maps_softmax = torch.softmax(maps.view(num_images*num_tokens, h * w), dim=-1)

    # Compute the entropy of each token
    entropy = dist.Categorical(probs=attention_maps_softmax).entropy()
    
    
    entropy = entropy.reshape(num_images, num_tokens)
    
    entropy = entropy.sum(dim=0)

    # get a sorted list of tokens with lowest entropy
    sorted_entropy = torch.argsort(entropy)
    
    chosen_maps = maps[:, sorted_entropy[:num_points], :, :]
    
    highest_indices = find_max_pixel(chosen_maps.view(num_images* num_points, h, w))
    
    highest_indices = highest_indices.reshape(num_images, num_points, 2)
    
    return highest_indices, sorted_entropy[:num_points]
