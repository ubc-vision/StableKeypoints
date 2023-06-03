from glob import glob
import torch
from tqdm import tqdm


{'pck': [37.5], 'opt_iterations_results': [[25.0], [37.5], [37.5], [37.5], [37.5]], 'inf_iterations_results': [[0.0], [0.0], [0.0], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5], [37.5]], 'ind_layers_results': [[12.5], [25.0], [0.0], [0.0]]}

pck = []
opt_iterations_results = []
inf_iterations_results = []
ind_layers_results = []

for file_name in tqdm(glob("/home/iamerich/burst/ldm/retesting/*")):
    file = torch.load(file_name)
    
    # import ipdb; ipdb.set_trace()
    
    pck.append(torch.tensor(file['pck']))
    opt_iterations_results.append(torch.tensor(file['opt_iterations_results']))
    inf_iterations_results.append(torch.tensor(file['inf_iterations_results']))
    ind_layers_results.append(torch.tensor(file['ind_layers_results']))
    
pck = torch.stack(pck)
opt_iterations_results = torch.stack(opt_iterations_results)
inf_iterations_results = torch.stack(inf_iterations_results)
ind_layers_results = torch.stack(ind_layers_results)

pck = torch.mean(pck, dim=0)
opt_iterations_results = torch.mean(opt_iterations_results, dim=0)
inf_iterations_results = torch.mean(inf_iterations_results, dim=0)
ind_layers_results = torch.mean(ind_layers_results, dim=0)

print("pck")
print(pck)
print("opt_iterations_results")
print(opt_iterations_results)
print("inf_iterations_results")
print(inf_iterations_results)
print("ind_layers_results")
print(ind_layers_results)

# make a plot for each of the results   
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

plt.plot(pck)
plt.title("pck")
plt.savefig("pck.png")
plt.close()

plt.plot(opt_iterations_results)
plt.title("number of optimization iterations")
plt.xticks(np.arange(0, 5, 1.0), np.arange(1, 6, 1.0).astype(int))
plt.savefig("opt_iterations_results.png")
plt.close()

plt.plot(inf_iterations_results[3:])
plt.title("number of inference iterations")
plt.xticks(np.arange(0, 17, 1.0), np.arange(4, 21, 1.0).astype(int))

plt.savefig("inf_iterations_results.png")
plt.close()

plt.plot(ind_layers_results)
plt.title("individual layers results")
plt.xticks(np.arange(0, 4, 1.0), np.arange(7, 11, 1.0).astype(int))
plt.savefig("ind_layers_results.png")
plt.close()
