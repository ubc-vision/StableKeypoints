import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils_training.utils import flow2kps, visualie_flow, visualie_correspondences
from utils_training.evaluation import Evaluator
from eval.keypoint_to_flow import KeypointToFlow
import ipdb

from networks.context_estimator import Context_Estimator

from optimize_token import optimize_prompt, find_max_pixel_value, visualize_image_with_points, run_image_with_tokens, find_context, softargmax2d, train_context_estimator, optimize_prompt_informed, optimize_prompt_faster, run_image_with_tokens_cropped, optimize_prompt_wo_gaussian

import wandb

import random

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
                   upsample_res = 512,
                   num_steps=100,
                   noise_level = 10,
                   layers = [0, 1, 2, 3, 4, 5],
                   num_words = 77,
                   epoch = 6,
                   device = 'cpu',
                   visualize = False,
                   wandb_log = False,
                   lr = 1e-3,
                   num_opt_iterations = 5,
                   sigma = 32,
                   flip_prob = 0.5,
                   crop_percent=80,
                   save_folder = "outputs",
                   item_index = -1,
                   alpha=0.1,
                   num_iterations=20,
                   ablate=False):
    

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    pck_array = []
    pck_array_ind_layers = [[] for i in range(len(layers))]
    for i, mini_batch in pbar:
        
        est_keypoints = -1*torch.ones_like(mini_batch['src_kps'])
        ind_layers = -1*torch.ones_like(mini_batch['src_kps']).repeat(len(layers), 1, 1)
        
        if ablate:
            # only evaluate the correspondence for the first point
            
            mini_batch['n_pts'] = torch.ones(1).int()
            mini_batch['src_kps'][0, 0, 1] = -1
            
        all_contexts = []

        
        for j in range(mini_batch['src_kps'].shape[2]):
            
            if mini_batch['src_kps'][0, 0, j] == -1:
                break
            
            if visualize:
                visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j], f"{i:03d}_initial_point_{j:02d}", save_folder=save_folder)
                visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'][0, :, j], f"{i:03d}_target_point_{j:02d}", save_folder=save_folder)
        
            contexts = []
                
            for _ in range(num_opt_iterations):
                
                # context = optimize_prompt(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=upsample_res, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                context = optimize_prompt_wo_gaussian(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=16, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                # context = optimize_prompt_faster(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=upsample_res, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                # context = optimize_prompt_informed(ldm, mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=upsample_res, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                contexts.append(context)
                    
            all_contexts.append(torch.stack(contexts))
            
            all_maps = []
            
            for context in contexts:
                
                maps = []
            
                attn_maps, _collected_attention_maps = run_image_with_tokens_cropped(ldm, mini_batch['og_trg_img'][0], context, index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations=num_iterations, image_mask = None if 'bool_img_trg' not in mini_batch else mini_batch['bool_img_trg'][0])
                
                for k in range(attn_maps.shape[0]):
                    avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                    maps.append(avg)
                    _max_val = find_max_pixel_value(avg[0], img_size = 512)
                    ind_layers[k, :, j] = (_max_val+0.5)
                    
                    # argmax = softargmax2d(avg)

                        # visualize_image_with_points(avg, argmax[0]*upsample_res, f"largest_loc_trg_softargmax_{j:02d}_{k:02d}")
                
                maps = torch.stack(maps, dim=0)
                
                all_maps.append(maps)
                
            all_maps = torch.stack(all_maps, dim=0)
            # take the average along dim=0
            all_maps = torch.mean(all_maps, dim=0)
            
            all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res*upsample_res))
            
            all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
            
            
            if visualize:
                for k in range(all_maps.shape[0]):
                    visualize_image_with_points(all_maps[k, None], mini_batch['trg_kps'][0, :, j]/512*upsample_res, f"{i:03d}_largest_loc_trg_{j:02d}_{k:02d}", save_folder=save_folder)
                
                
            # all_maps = torch.max(all_maps, dim=0).values
            all_maps = torch.mean(all_maps, dim=0)
                
            max_val = find_max_pixel_value(all_maps, img_size = 512)
            
            # # import ipdb; ipdb.set_trace()
            
            est_keypoints[0, :, j] = (max_val+0.5)
            
            
            if visualize:
                
                all_maps = []
                
                for context in contexts:
                    # visualize_image_with_points(mini_batch['og_trg_img'][0], (max_val+0.5), f"largest_loc_trg_img_{j:02d}")
                    
                    attn_map_src, _collected_attention_maps = run_image_with_tokens_cropped(ldm, mini_batch['og_src_img'][0], context, index=0, upsample_res=upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations=num_iterations, image_mask = None if 'bool_img_src' not in mini_batch else mini_batch['bool_img_src'][0])
                    
                    maps = []
                    for k in range(attn_map_src.shape[0]):
                        avg = torch.mean(attn_map_src[k], dim=0, keepdim=True)
                        maps.append(avg)
                        
                    maps = torch.stack(maps, dim=0)
                
                    all_maps.append(maps)
                    
                all_maps = torch.stack(all_maps, dim=0)
                # take the average along dim=0
                all_maps = torch.mean(all_maps, dim=0)
                
                all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res*upsample_res))
            
                all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
                        
                        
                for k in range(all_maps.shape[0]):  
                    visualize_image_with_points(all_maps[k, None], mini_batch['src_kps'][0, :, j]/512*upsample_res, f"{i:03d}_largest_loc_src_{j:02d}_{k:02d}", save_folder=save_folder)

                # visualize_image_with_points(mini_batch['og_src_img'][0], (max_val+0.5), f"largest_loc_src_img_{j:02d}")
                
                # exit()
            
        for k in range(len(pck_array_ind_layers)):
            _eval_result = Evaluator.eval_kps_transfer(ind_layers[k].cpu()[None], mini_batch)
            pck_array_ind_layers[k] += _eval_result['pck']
            
            print(f"layer {k} pck {sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])}, this pck {_eval_result['pck']}")

        

        eval_result = Evaluator.eval_kps_transfer(est_keypoints.cpu(), mini_batch)
        
        
        if visualize:
            visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], est_keypoints, f"correspondences_estimated_{i:03d}", correct_ids = eval_result['correct_ids'], save_folder=save_folder)
            visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], mini_batch['trg_kps'], f"correspondences_gt_{i:03d}", correct_ids = eval_result['correct_ids'], save_folder=save_folder)
            
        dict = {"est_keypoints": est_keypoints, "correct_ids": eval_result['correct_ids'], "src_kps": mini_batch['src_kps'], "trg_kps": mini_batch['trg_kps'], "idx": mini_batch['idx'] if item_index == -1 else item_index, "contexts": torch.stack(all_contexts), 'pck': eval_result['pck']}
        # save dict 
        torch.save(dict, f"{save_folder}/correspondence_data_{i:03d}.pt")

        pck_array += eval_result['pck']

        mean_pck_ten = sum(pck_array[1::2]) / len(pck_array[1::2])
        
        
        print(f"epoch: {epoch} {i} this pck ", eval_result['pck'], " mean_pck " , mean_pck_ten)
        # exit()
        
        if wandb_log:
            wandb_dict = {"pck": mean_pck_ten}
            for k in range(len(pck_array_ind_layers)):
                wandb_dict[f"pck_layer_{k}"] = sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])
            wandb.log(wandb_dict)

    return pck_array



def retest(ldm,
            test_dataset,
            upsample_res = 512,
            num_steps=100,
            noise_level = 10,
            layers = [0, 1, 2, 3, 4, 5],
            num_words = 77,
            epoch = 6,
            crop_percent=100.0,
            device = 'cpu',
            visualize = False,
            optimize = False,
            wandb_log = False,
            item_index = -1,
            ablate_results = False,
            alpha=0.1,
            num_iterations = 20,
            results_loc = "/scratch/iamerich/prompt-to-prompt/outputs/ldm_visualization_020",
            save_folder = "outputs"):
    """
    if ablate_results:
        saves performance for average of 1-5 optimization iterations
        saves performance for average of 1-20 inference iterations
        saves performance of each layer
    """
    
    from glob import glob
    import re
    
    files = glob(f"{results_loc}/*/correspondence_data_*.pt")
    
    correspondences = sorted(files, key=lambda path: int(re.search(r'/(\d+)/', path).group(1)))

    
    # correspondences = sorted(glob(f"{results_loc}/*/correspondence_data_*.pt"))
    # print("len(correspondences)")
    # print(len(correspondences))
    
    if item_index != -1:
        correspondences = [correspondences[item_index]]

    pck_array = []
    pck_array_ind_layers = [[] for _ in range(len(layers))]
    # for i, correspondence in enumerate(correspondences):
    for i in range(len(correspondences)):
        
        data = torch.load(correspondences[i], map_location=device)
        
        contexts = data["contexts"].to(device)
        
        idx = data['idx']
        
        mini_batch = test_dataset[idx]
        
        mini_batch['pckthres'] = mini_batch['pckthres'][None]
        mini_batch['n_pts'] = mini_batch['n_pts'][None]
        mini_batch['trg_kps'] = mini_batch['trg_kps'][None]
        
        est_keypoints = -1*torch.ones_like(mini_batch['src_kps'])
        ind_layers = -1*torch.ones_like(mini_batch['src_kps']).repeat(len(layers), 1, 1)
        ind_opt_iterations = -1*torch.ones_like(mini_batch['src_kps']).repeat(10, 1, 1)
        ind_inf_iterations = -1*torch.ones_like(mini_batch['src_kps']).repeat(num_iterations, 1, 1)
        
        # import ipdb; ipdb.set_trace()
        
        # for j in [mini_batch['src_kps'].shape[1]-1]:
        for j in range(contexts.shape[0]):
            print(j)
        # for _ in range(1):
            
            assert mini_batch['src_kps'][0, j] != -1
            
            if visualize:
                visualize_image_with_points(mini_batch['og_src_img'], mini_batch['src_kps'][:, j], f"{i:03d}_initial_point_{j:02d}", save_folder=save_folder)
                visualize_image_with_points(mini_batch['og_trg_img'], mini_batch['trg_kps'][0, :, j], f"{i:03d}_target_point_{j:02d}", save_folder=save_folder)
            
            all_maps = []
            
            collected_attention_maps = []
            
            # for context in contexts:
            for l in range(contexts.shape[1]):
                
                maps = []
            
                # attn_maps = run_image_with_tokens(ldm, mini_batch['og_trg_img'], contexts[j, l].to(device), index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers, device=device)
                attn_maps, _collected_attention_maps = run_image_with_tokens_cropped(ldm, mini_batch['og_trg_img'], contexts[j, l].to(device), index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations = num_iterations)
                
                collected_attention_maps.append(torch.stack(_collected_attention_maps, dim=0).detach().cpu())
                
                for k in range(attn_maps.shape[0]):
                    avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                    maps.append(avg)
                    assert avg.shape[0] == 1
                    _max_val = find_max_pixel_value(avg[0], img_size = 512)
                    ind_layers[k, :, j] = (_max_val+0.5)
                    
                    # argmax = softargmax2d(avg)

                        # visualize_image_with_points(avg, argmax[0]*upsample_res, f"largest_loc_trg_softargmax_{j:02d}_{k:02d}")
                
                maps = torch.stack(maps, dim=0)
                
                all_maps.append(maps)
                
                if ablate_results:
                    
                    # save performance per optimization iteration
                    mean_this_it = torch.mean(torch.stack(all_maps, dim=0), dim=0)
                    # all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res*upsample_res))
                    # all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
                    mean_this_it = torch.mean(mean_this_it, dim=0)
                    assert mean_this_it.shape[0] == 1
                    _max_val = find_max_pixel_value(mean_this_it[0], img_size = 512)
                    # # import ipdb; ipdb.set_trace()
                    
                    ind_opt_iterations[l, :, j] = (_max_val+0.5)
                    
            if ablate_results:
                collected_attention_maps = torch.stack(collected_attention_maps, dim=1)
                for k in range(collected_attention_maps.shape[0]):
                    mean_this_it = torch.mean(collected_attention_maps[k], dim=0)
                    mean_this_it = torch.mean(mean_this_it, dim=0)
                    mean_this_it = torch.mean(mean_this_it, dim=0)
                    _max_val = find_max_pixel_value(mean_this_it, img_size = 512)
                    ind_inf_iterations[k, :, j] = _max_val+0.5
                
            all_maps = torch.stack(all_maps, dim=0)
            # take the average along dim=0
            all_maps = torch.mean(all_maps, dim=0)
            
            # all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res*upsample_res))
            
            all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
            
            
            if visualize:
                for k in range(all_maps.shape[0]):
                    visualize_image_with_points(all_maps[k, None], mini_batch['trg_kps'][0, :, j]/512*upsample_res, f"{i:03d}_largest_loc_trg_{j:02d}_{k:02d}", save_folder=save_folder)
                
                
            # all_maps = torch.max(all_maps, dim=0).values
            all_maps = torch.mean(all_maps, dim=0)
                
            max_val = find_max_pixel_value(all_maps, img_size = 512)
            
            # # import ipdb; ipdb.set_trace()
            
            est_keypoints[:, j] = (max_val+0.5)
            
            
            if visualize:
                
                all_maps = []
                
                for l in range(contexts.shape[1]):
                    # visualize_image_with_points(mini_batch['og_trg_img'][0], (max_val+0.5), f"largest_loc_trg_img_{j:02d}")
                    
                    attn_map_src, _ = run_image_with_tokens_cropped(ldm, mini_batch['og_src_img'], contexts[j, l], index=0, upsample_res=upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations=num_iterations)
                    
                    maps = []
                    for k in range(attn_map_src.shape[0]):
                        avg = torch.mean(attn_map_src[k], dim=0, keepdim=True)
                        maps.append(avg)
                        
                    maps = torch.stack(maps, dim=0)
                
                    all_maps.append(maps)
                    
                all_maps = torch.stack(all_maps, dim=0)
                # take the average along dim=0
                all_maps = torch.mean(all_maps, dim=0)
                
                # all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res*upsample_res))
            
                all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
                        
                        
                for k in range(all_maps.shape[0]):  
                    visualize_image_with_points(all_maps[k, None], mini_batch['src_kps'][:, j]/512*upsample_res, f"{i:03d}_largest_loc_src_{j:02d}_{k:02d}", save_folder=save_folder)
                
        ind_layers_results = []
            
        for k in range(ind_layers.shape[0]):
            _eval_result = Evaluator.eval_kps_transfer(ind_layers[k].cpu()[None], mini_batch)
            pck_array_ind_layers[k] += _eval_result['pck']
            
            ind_layers_results.append(_eval_result['pck'])
            
            print(f"layer {k} pck {sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])}, this pck {_eval_result['pck']}")
            
        if ablate_results:
            opt_iterations_results = []    
            for k in range(ind_opt_iterations.shape[0]):
                _eval_result = Evaluator.eval_kps_transfer(ind_opt_iterations[k].cpu()[None], mini_batch)
                opt_iterations_results.append(_eval_result['pck'])
                
            inf_iterations_results = []    
            for k in range(ind_inf_iterations.shape[0]):
                _eval_result = Evaluator.eval_kps_transfer(ind_inf_iterations[k].cpu()[None], mini_batch)
                inf_iterations_results.append(_eval_result['pck'])
        

        eval_result = Evaluator.eval_kps_transfer(est_keypoints[None].cpu(), mini_batch)
        
        
        if visualize:
            visualie_correspondences(mini_batch['og_src_img'], mini_batch['og_trg_img'], mini_batch['src_kps'][None], est_keypoints[None], f"correspondences_estimated_{i:03d}", correct_ids = eval_result['correct_ids'], save_folder=save_folder)
            visualie_correspondences(mini_batch['og_src_img'], mini_batch['og_trg_img'], mini_batch['src_kps'][None], mini_batch['trg_kps'], f"correspondences_gt_{i:03d}", correct_ids = eval_result['correct_ids'], save_folder=save_folder)

        pck_array += eval_result['pck']

        mean_pck = sum(pck_array) / len(pck_array)
        
        
        print(f"epoch: {epoch} {i} this pck ", eval_result['pck'], " mean_pck " , mean_pck)
        # exit()
        
        if wandb_log:
            wandb_dict = {"pck": mean_pck}
            for k in range(len(pck_array_ind_layers)):
                wandb_dict[f"pck_layer_{k}"] = sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])
            wandb.log(wandb_dict)
            
        if ablate_results:
            torch.save({
                'pck': eval_result['pck'],
                'opt_iterations_results': opt_iterations_results,
                'inf_iterations_results': inf_iterations_results,
                'ind_layers_results': ind_layers_results
            }, f"{save_folder}/{idx:06d}_results.pt")

    return pck_array

def rewrite_idxs():
    
    from glob import glob
    
    correspondences = sorted(glob("/scratch/iamerich/prompt-to-prompt/outputs/ldm_visualization_020/*/correspondence_data_*.pt"))
    
    for i in tqdm(range(len(correspondences))):
        
        data = torch.load(correspondences[i])
        
        data['idx'] = i
        
        # save the data
        torch.save(data, correspondences[i])
        
        