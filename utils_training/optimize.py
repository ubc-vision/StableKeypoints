import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils_training.utils import flow2kps, visualie_flow, visualize_image_with_points, visualie_correspondences
from utils_training.evaluation import Evaluator
from eval.keypoint_to_flow import KeypointToFlow
import ipdb

from optimize_token import optimize_prompt, find_max_pixel_value, visualize_image_with_points, run_image_with_tokens

r'''
    loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)


def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer):
    n_iter = epoch*len(train_loader)
    
    net.train()
    running_total_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        flow_gt = mini_batch['flow'].to(device)

        pred_flow = net(mini_batch['trg_img'].to(device),
                         mini_batch['src_img'].to(device))
        
        Loss = EPE(pred_flow, flow_gt) 
        Loss.backward()
        optimizer.step()

        running_total_loss += Loss.item()
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
    running_total_loss /= len(train_loader)
    return running_total_loss

def save_img(img, save_path):
    from PIL import Image
    img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f"outputs/{save_path}")


def validate_epoch(ldm,
                   val_loader,
                   device,
                   epoch,
                   upsample_res = 512,
                   num_steps=100,
                   noise_level = 10,
                   layers = [0, 1, 2, 3, 4, 5],
                   num_words = 77):
    running_total_loss = 0
    
    
    


    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    pck_array = []
    pck_array_ind_layers = [[] for i in range(len(layers))]
    for i, mini_batch in pbar:
        
        est_keypoints = -1*torch.ones_like(mini_batch['src_kps'])
        ind_layers = -1*torch.ones_like(mini_batch['src_kps']).repeat(len(layers), 1, 1)
        
        

        
        for j in range(mini_batch['src_kps'].shape[2]):
            
            if mini_batch['src_kps'][0, 0, j] == -1:
                continue
            
            
            # visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j], f"initial_point_{j:02d}")
            # visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'][0, :, j], f"target_point_{j:02d}")
            
            # print("optimizing prompt")
            context = optimize_prompt(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, target_image = mini_batch['og_trg_img'][0], context=None, device="cuda", num_steps=num_steps, upsample_res=upsample_res, noise_level=noise_level, layers=layers, bbox_initial=mini_batch['src_bbox_og'], bbox_target=mini_batch['trg_bbox_og'], num_words=num_words)
            
            # print("running token")
            
            attn_maps = run_image_with_tokens(ldm, mini_batch['og_trg_img'][0], context, index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers)
            
            
            
            
            maps = []
            for k in range(attn_maps.shape[0]):
                avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                maps.append(avg.reshape(-1))
                _max_val = find_max_pixel_value(avg[0], img_size = 512)
                ind_layers[k, :, j] = (_max_val+0.5)
                # print('_max_val')
                # print(_max_val)
                # print('attn_maps[k].shape')
                # print(attn_maps[k].shape)
                # visualize_image_with_points(avg, _max_val/512*upsample_res, f"largest_loc_trg_{j:02d}_{k:02d}")
            
            # import ipdb; ipdb.set_trace()
            
            maps = torch.stack(maps, dim=0)
            maps = torch.nn.Softmax(dim=-1)(maps)
            
            maps = torch.max(maps, dim=0).values
            maps = maps.reshape(upsample_res, upsample_res)
            max_val = find_max_pixel_value(maps, img_size = 512)
            
            # import ipdb; ipdb.set_trace()
            
            est_keypoints[0, :, j] = (max_val+0.5)
            
            # visualize_image_with_points(mini_batch['og_trg_img'][0], (max_val+0.5), f"largest_loc_trg_img_{j:02d}")
            
            
            
            # attn_map_src = run_image_with_tokens(ldm, mini_batch['og_src_img'][0], context, index=0, upsample_res=upsample_res, noise_level=noise_level, layers=layers)
            
            # maps = []
            # for k in range(attn_map_src.shape[0]):
            #     avg = torch.mean(attn_map_src[k], dim=0, keepdim=True)
            #     maps.append(avg)
                
            #     max_val_src = find_max_pixel_value(avg[0], img_size = 512)
            #     visualize_image_with_points(avg, max_val_src/512*upsample_res, f"largest_loc_src_{j:02d}_{k:02d}")
            # maps = torch.cat(maps, dim=0)
            # maps = torch.mean(maps, dim=0)
            # max_val = find_max_pixel_value(maps, img_size = 512)
            # visualize_image_with_points(mini_batch['og_src_img'][0], (max_val+0.5), f"largest_loc_src_img_{j:02d}")
            
            # exit()
            
        for k in range(len(pck_array_ind_layers)):
            _eval_result = Evaluator.eval_kps_transfer(ind_layers[k].cpu()[None], mini_batch)
            pck_array_ind_layers[k] += _eval_result['pck']
            
            print(f"layer {k} pck {sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])}, this pck {_eval_result['pck']}")
        
        

        

        eval_result = Evaluator.eval_kps_transfer(est_keypoints.cpu(), mini_batch)
        
        
        
        visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], est_keypoints, f"correspondences_estimated_{i:03d}", correct_ids = eval_result['correct_ids'])
        visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], mini_batch['trg_kps'], f"correspondences_gt_{i:03d}", correct_ids = eval_result['correct_ids'])

        pck_array += eval_result['pck']

        mean_pck = sum(pck_array) / len(pck_array)
        
        
        print(f"{i} this pck ", eval_result['pck'], " mean_pck " , mean_pck)

    return running_total_loss / len(val_loader), mean_pck