import torch
import torch.distributions as dist

from scipy.stats import kurtosis

num_tokens = 1

image = torch.ones((num_tokens, 10))*0.0

image[:, 5] = 0.01

# randomize values for image between 0 and 1
# image = torch.rand((num_tokens, 10))

# Normalize the activation maps to represent probability distributions
attention_maps_softmax = torch.softmax(image.view(num_tokens, -1), dim=-1)

# Compute the entropy of each token
entropy = dist.Categorical(probs=attention_maps_softmax.view(num_tokens, -1)).entropy()

print("entropy")
print(entropy)

# kurt = torch.tensor([kurtosis(img.view(num_tokens, -1).detach().cpu().numpy()) for img in image])

# print("kurt")
# print(kurt)
