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

from optimize_token import optimize_prompt, find_max_pixel_value, visualize_image_with_points, run_image_with_tokens, find_context, softargmax2d, train_context_estimator, optimize_prompt_informed, optimize_prompt_faster

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


def train(ldm,
            val_loader,
            upsample_res = 512,
            num_steps=100,
            noise_level = 10,
            layers = [0, 1, 2, 3, 4, 5],
            num_words = 77,
            wandb_log = False,
            device = "cuda",
            save_loc = "outputs",
            learning_rate = 1e-4,):
    running_total_loss = 0
    
    # ldm = torch.nn.DataParallel(ldm)
    
    context_estimator = Context_Estimator(ldm.tokenizer, ldm.text_encoder, num_words = num_words, device=device).cuda()
    
    # context_estimator = torch.nn.DataParallel(context_estimator)
    
    optimizer = torch.optim.Adam(context_estimator.parameters(), lr=learning_rate)
    
    
    for epoch in range(100):

        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        # pck_array = []
        # pck_array_ind_layers = [[] for i in range(len(layers))]
        for i, mini_batch in pbar:
            
            # est_keypoints = -1*torch.ones_like(mini_batch['src_kps'])
            # ind_layers = -1*torch.ones_like(mini_batch['src_kps']).repeat(len(layers), 1, 1)
            
            
            # select an index from mini_batch['src_kps'][0, 0, :] that is not -1
            non_negative_one = torch.where(mini_batch['src_kps'][0, 0, :] != -1)[0]
            random_non_negative_one = torch.where(mini_batch['random_kps'][0, 0, :] != -1)[0]
            
            j = random.randint(0, non_negative_one.shape[0]-1)
            k = random.randint(0, random_non_negative_one.shape[0]-1)
            
            # for j in range(mini_batch['src_kps'].shape[2]):
                
            #     if mini_batch['src_kps'][0, 0, j] == -1:
            #         continue
            
            
            
            
            
            context = train_context_estimator(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, context_estimator, optimizer, target_image = mini_batch['og_trg_img'][0], random_image = mini_batch['og_random_img'][0], rand_img_keypoint=mini_batch['random_kps'][0, :, k]/512, context=None, device="cuda", num_steps=num_steps, upsample_res=upsample_res, noise_level=noise_level, layers=layers, bbox_initial=mini_batch['src_bbox_og'], bbox_target=mini_batch['trg_bbox_og'], bbox_random = mini_batch['random_bbox_og'], num_words=num_words, wandb_log=wandb_log)
                
                # attn_maps = run_image_with_tokens(ldm, mini_batch['og_trg_img'][0], context, index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers)
                
                
                # maps = []
                # for k in range(attn_maps.shape[0]):
                #     avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                #     maps.append(avg.reshape(-1))
                #     _max_val = find_max_pixel_value(avg[0], img_size = 512)
                #     ind_layers[k, :, j] = (_max_val+0.5)
                
                # maps = torch.stack(maps, dim=0)
                # maps = torch.nn.Softmax(dim=-1)(maps)
                
                # maps = torch.max(maps, dim=0).values
                # maps = maps.reshape(upsample_res, upsample_res)
                # max_val = find_max_pixel_value(maps, img_size = 512)
                
                # # import ipdb; ipdb.set_trace()
                
                # est_keypoints[0, :, j] = (max_val+0.5)
                
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
                
            # for k in range(len(pck_array_ind_layers)):
            #     _eval_result = Evaluator.eval_kps_transfer(ind_layers[k].cpu()[None], mini_batch)
            #     pck_array_ind_layers[k] += _eval_result['pck']
                
            #     print(f"layer {k} pck {sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])}, this pck {_eval_result['pck']}")
            
            

            

            # eval_result = Evaluator.eval_kps_transfer(est_keypoints.cpu(), mini_batch)
            
            
            
            # visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], est_keypoints, f"correspondences_estimated_{i:03d}", correct_ids = eval_result['correct_ids'])
            # visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], mini_batch['trg_kps'], f"correspondences_gt_{i:03d}", correct_ids = eval_result['correct_ids'])

            # pck_array += eval_result['pck']

            # mean_pck = sum(pck_array) / len(pck_array)
            
            
            # print(f"epoch: {epoch} {i} this pck ", eval_result['pck'], " mean_pck " , mean_pck)
            
        # save context_estimator
        torch.save(context_estimator.state_dict(), f"{save_loc}/context_estimator_{epoch:03d}.pt")

    return running_total_loss / len(val_loader), mean_pck


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
                   optimize = False,
                   wandb_log = False,
                   lr = 1e-3,
                   num_iterations = 5,
                   sigma = 32,
                   flip_prob = 0.5,
                   crop_percent=80,
                   save_folder = "outputs",
                   ablate=False):
    
    
    if not optimize:
        context_estimator = Context_Estimator(ldm.tokenizer, ldm.text_encoder, num_words = num_words, device = device)
        context_estimator.load_state_dict(torch.load(f"checkpoints/context_estimator_{epoch:03d}.pt"))
    
    

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

        
        for j in range(mini_batch['src_kps'].shape[2]):
            
            if mini_batch['src_kps'][0, 0, j] == -1:
                break
            
            if visualize:
                visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j], f"{i:03d}_initial_point_{j:02d}", save_folder=save_folder)
                visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'][0, :, j], f"{i:03d}_target_point_{j:02d}", save_folder=save_folder)
        
            contexts = []

        
            if not optimize:
                # context = optimize_prompt(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, context_estimator, optimizer, target_image = mini_batch['og_trg_img'][0], context=None, device="cuda", num_steps=num_steps, upsample_res=upsample_res, noise_level=noise_level, layers=layers, bbox_initial=mini_batch['src_bbox_og'], bbox_target=mini_batch['trg_bbox_og'], num_words=num_words, wandb_log=wandb_log)
                
                context = find_context(mini_batch['og_src_img'][0], ldm, mini_batch['src_kps'][0, :, j]/512, context_estimator, device=device, )
                contexts.append(context)
            else:
                
                for _ in range(num_iterations):
                    
                    print("iterating")
                    context = optimize_prompt(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=upsample_res, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                    # context = optimize_prompt_faster(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=upsample_res, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                    # context = optimize_prompt_informed(ldm, mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=upsample_res, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                    contexts.append(context)
            
            all_maps = []
            
            for context in contexts:
                
                maps = []
            
                attn_maps = run_image_with_tokens(ldm, mini_batch['og_trg_img'][0], context, index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers, device=device)
                
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
                    
                    attn_map_src = run_image_with_tokens(ldm, mini_batch['og_src_img'][0], context, index=0, upsample_res=upsample_res, noise_level=noise_level, layers=layers, device=device)
                    
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
            dict = {"est_keypoints": est_keypoints, "correct_ids": eval_result['correct_ids'], "src_kps": mini_batch['src_kps'], "trg_kps": mini_batch['trg_kps'], "idx": mini_batch['idx'], "contexts": torch.stack(contexts)}
            # save dict 
            torch.save(dict, f"{save_folder}/correspondence_data_{i:03d}.pt")

        pck_array += eval_result['pck']

        mean_pck = sum(pck_array) / len(pck_array)
        
        
        print(f"epoch: {epoch} {i} this pck ", eval_result['pck'], " mean_pck " , mean_pck)
        # exit()
        
        if wandb_log:
            wandb_dict = {"pck": mean_pck}
            for k in range(len(pck_array_ind_layers)):
                wandb_dict[f"pck_layer_{k}"] = sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])
            wandb.log(wandb_dict)

    return pck_array

