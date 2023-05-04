# load all files in /home/iamerich/burst/cubs_unflipped/
import os
import torch
from tqdm import tqdm
from eval import download
from torch.utils.data import DataLoader
from utils_training.evaluation import Evaluator




test_dataset = download.load_dataset("cubs", "/home/iamerich/burst/Datasets_CATs", 0.1, 'cpu',
                                            "test", False, 16, sub_class=None, item_index=-1)

test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=4,
                                 shuffle=False)

loader = iter(test_dataloader)

pck_five = []
pck_ten = []

diff = 0

for i in tqdm(range(len(test_dataloader))):

    mini_batch = next(loader)
    
    # if mini_batch['og_trg_img_size'] < 300 or mini_batch['og_src_img_size'] < 300:
    #     continue
    
    # print("mini_batch['og_trg_img_size']")
    # print(mini_batch['og_trg_img_size'])
    # print("mini_batch['pckthres']")
    # print(mini_batch['pckthres'])
    
    diff += mini_batch['pckthres']-mini_batch['og_trg_img_size']

    
    mini_batch['pckthres'] = mini_batch['og_trg_img_size']
    
    # if the folder doesnt exist, pass
    if not os.path.exists(f"/home/iamerich/burst/cubs_unflipped/{i}") or not os.path.exists(f"/home/iamerich/burst/cubs_unflipped/{i}/correspondence_data_000.pt"):
        continue
    
    est_keypoints = torch.load(f"/home/iamerich/burst/cubs_unflipped/{i}/correspondence_data_000.pt", map_location='cpu')['est_keypoints']

    eval_result = Evaluator.eval_kps_transfer(est_keypoints.cpu(), mini_batch)
    
    # print(eval_result)
    pck_five.append(eval_result['pck'][0])
    pck_ten.append(eval_result['pck'][1])
    # exit()
    
print(sum(pck_five)/len(pck_five))
print(sum(pck_ten)/len(pck_ten))

print(diff/len(pck_five))
