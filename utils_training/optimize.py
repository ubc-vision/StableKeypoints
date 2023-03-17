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
                   epoch):
    running_total_loss = 0


    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    pck_array = []
    for i, mini_batch in pbar:
        
        # try:
        
        # if i < 5125:
        #     continue
        
        # print("mini_batch['src_kps']")
        # print(mini_batch['src_kps'])
        
        est_keypoints = -1*torch.ones_like(mini_batch['src_kps'])
        
        # # print("mini_batch['src_kps'].shape")
        # # print(mini_batch['src_kps'].shape)
        
        # mask = mini_batch['src_kps'][0, 0, :] != -1
        
        # # for i in range(mask.shape[0]):
        # #     if mask[i] == False:
        # #         continue
        # #     visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, i], f"source_img_{i:02d}_gt")
        # #     visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'][0, :, i], f"target_img_{i:02d}_gt")
        
        # this_context = optimize_prompt_over_subject(ldm, mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'][0][:, mask]/512, num_steps=100, device=device)
        
        
        # visualize_keypoints_over_subject(ldm, mini_batch['og_src_img'][0], this_context, "source_img", device=device)
        # trg_attn = visualize_keypoints_over_subject(ldm, mini_batch['og_trg_img'][0], this_context, "target_img", device=device)
        
        # # print("trg_attn.shape")
        # # print(trg_attn.shape)
        # # exit()
        
        # for j in range(this_context.shape[0]):
        #     # print("attn.shape")
        #     # print(attn.shape)
        #     max_val = find_max_pixel_value(trg_attn[j])
        #     # print("max_val")
        #     # print(max_val)
            
        #     est_keypoints[0, :, j] = (max_val+0.5)*512/16
        # # exit()
        
        
        
        # for i in range(this_context.shape[0]):
            
        #     visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, i], f"initial_point_{i:02d}")
        #     visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'][0, :, i], f"target_point_{i:02d}")
        
        #     attn_map = find_average_attention_from_list(mini_batch['og_trg_img'][0], ldm, [this_context[i]], f"attn_trg_{i:02d}", device=device, index=0)
        #     max_val = find_max_pixel_value(attn_map)
            
        #     est_keypoints[0, :, i] = (max_val+0.5)*512/16
            
        #     visualize_image_with_points(attn_map[None], max_val, f"largest_loc_trg_{i:02d}")
        #     visualize_image_with_points(mini_batch['og_trg_img'][0], (max_val+0.5)*512/16, f"largest_loc_trg_img_{i:02d}")
            
            
        #     attn_map = find_average_attention_from_list(mini_batch['og_src_img'][0], ldm, [this_context[i]], f"attn_src_{i:02d}", device=device, index=0)
        #     max_val = find_max_pixel_value(attn_map)
        #     visualize_image_with_points(attn_map[None], max_val, f"largest_loc_src_{i:02d}")
        #     visualize_image_with_points(mini_batch['og_src_img'][0], (max_val+0.5)*512/16, f"largest_loc_src_img_{i:02d}")
        
        # exit()

        
        
        
        # # make the learned contexts as similar as possible accross mini_batch['src_kps']
        # for j in range(mini_batch['src_kps'].shape[2]):
            
        #     if mini_batch['src_kps'][0, 0, j] == -1:
        #         continue
            
        #     visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j], f"initial_point_{j:02d}")
        #     visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'][0, :, j], f"target_point_{j:02d}")
            
            
            
            
        #     contexts = []
        #     for i in range(10):
        #         # this_context = optimize_prompt(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=100, device=device)
        #         this_context = optimize_prompt_informed(ldm, mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=150, device=device)
        #         contexts.append(this_context.detach())
                
                
                
        #     attn_map = find_average_attention_from_list(mini_batch['og_trg_img'][0], ldm, contexts, f"attn_trg_{j:02d}", device=device, index=0)
        #     max_val = find_max_pixel_value(attn_map)
            
        #     est_keypoints[0, :, j] = (max_val+0.5)*512/32
            
        #     visualize_image_with_points(attn_map[None], max_val, f"largest_loc_trg_{j:02d}")
        #     visualize_image_with_points(mini_batch['og_trg_img'][0], (max_val+0.5)*512/32, f"largest_loc_trg_img_{j:02d}")
            
            
        #     attn_map = find_average_attention_from_list(mini_batch['og_src_img'][0], ldm, contexts, f"attn_src_{j:02d}", device=device, index=0)
        #     max_val = find_max_pixel_value(attn_map)
        #     visualize_image_with_points(attn_map[None], max_val, f"largest_loc_src_{j:02d}")
        #     visualize_image_with_points(mini_batch['og_src_img'][0], (max_val+0.5)*512/32, f"largest_loc_src_img_{j:02d}")
        #     # exit()
            
            
            
        # # est_keypoints = torch.cat(est_keypoints, dim=-1)
        
        # # print("est_keypoints")
        # # print(est_keypoints)
        # # print("mini_batch['trg_kps']")
        # # print(mini_batch['trg_kps'])
        
        # visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], est_keypoints, f"estimated_correspondences_{i:03d}")
        # exit()
        # except:
        #     from time import sleep
        #     sleep(0.25)
        #     continue
        
        # exit()
        
        
        
        for j in range(mini_batch['src_kps'].shape[2]):
            
            if mini_batch['src_kps'][0, 0, j] == -1:
                continue
            
            
            visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j], f"initial_point_{j:02d}")
            visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'][0, :, j], f"target_point_{j:02d}")
        
            print("optimizing prompt")
            context = optimize_prompt(ldm, mini_batch['og_src_img'][0], mini_batch['src_kps'][0, :, j]/512, context=None, device="cuda", num_steps=100)
            
            print("context.shape")
            print(context.shape)
            
            print("running token")
            
            attn_map = run_image_with_tokens(ldm, mini_batch['og_src_img'][0], context, index=0)
            
            max_val = find_max_pixel_value(attn_map)
            
            
            visualize_image_with_points(attn_map[None], max_val, f"largest_loc_trg_{j:02d}")
            visualize_image_with_points(mini_batch['og_trg_img'][0], (max_val+0.5), f"largest_loc_trg_img_{j:02d}")
            
            est_keypoints[0, :, j] = (max_val+0.5)
            
            attn_map_src = run_image_with_tokens(ldm, mini_batch['og_src_img'][0], context, index=0)
            max_val_src = find_max_pixel_value(attn_map_src)
            visualize_image_with_points(attn_map[None], max_val, f"largest_loc_src_{j:02d}")
            visualize_image_with_points(mini_batch['og_src_img'][0], (max_val+0.5), f"largest_loc_src_img_{j:02d}")
        
        visualie_correspondences(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], mini_batch['src_kps'], est_keypoints, f"estimated_correspondences_{i:03d}")
        
        # visualize_image_with_points(mini_batch['og_src_img'][0], mini_batch['src_kps'], 'src_kps')
        
        # visualize_image_with_points(mini_batch['og_trg_img'][0], mini_batch['trg_kps'], 'trg_kps')
        # exit()
        
        # visualie_flow(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], flow_gt[0], "flow")
        
        # print("mini_batch['trg_kps'].shape")
        # print(mini_batch['trg_kps'].shape)
        
        # print("flow_gt")
        # print(flow_gt)
        # print entire pytorch tensor without clipping      
        # torch.set_printoptions(threshold=10_000)
        
        # save_img(mini_batch['og_src_img'][0], "src_img.png")
        # save_img(mini_batch['og_trg_img'][0], "trg_img.png")
        
        # flow_gt_img = torch.cat([flow_gt[0], torch.zeros(1, flow_gt[0].shape[1], flow_gt[0].shape[2]).cuda()], dim=0)
        # save_img(flow_gt_img, "flow_gt.png")


        
        # print("flow_gt")
        # print(flow_gt)
        
        
        
        # # select nonzero entries of flow_gt
        # mask = (flow_gt[:,0] == 0) & (flow_gt[:,1] == 0)
        # print("mask.shape")
        # print(mask.shape)
        # flow_gt = flow_gt[~mask[:, None].repeat(1, 2, 1, 1)]
    

        # # print("flow_gt")
        # # print(flow_gt)
        # # exit()
        
        
        
        # print("mini_batch['trg_img'].shape")
        # print(mini_batch['trg_img'].shape)
        # print("flow_gt.shape")
        # print(flow_gt.shape)
        # print("flow_gt[0]")
        # print(flow_gt[0])
        # print("mini_batch['trg_img'].shape")
        # print(mini_batch['trg_img'].shape)
        # print("mini_batch['n_pts'].shape")
        # print(mini_batch['n_pts'].shape)
        # print("mini_batch['trg_kps'].shape")
        # print(mini_batch['trg_kps'].shape)
        # print("mini_batch['trg_kps'][0]")
        # print(mini_batch['trg_kps'][0])
        
        # # save mini_batch['trg_img'][0] as an image
        
        # from PIL import Image
        
        # print("torch.max(mini_batch['trg_img'])")
        # print(torch.max(mini_batch['trg_img']))
        # print("torch.min(mini_batch['trg_img'])")
        # print(torch.min(mini_batch['trg_img']))
        
        # save_img(mini_batch['og_trg_img'][0], "trg_img.png")
        # save_img(mini_batch['og_src_img'][0], "src_img.png")
        # save_img(flow_gt_img, "flow_gt.png")
        
        
        
        # print("mask.shape")
        # print(mask.shape)
        # print("torch.sum(mask)")
        # print(torch.sum(mask))
        # print("torch.sum(~mask)")
        # print(torch.sum(~mask))
        
        # pred_flow = net(mini_batch['trg_img'].to(device),
        #                 mini_batch['src_img'].to(device))
        # pred_flow = flow_gt.clone()
        
        # pred_flow_img = torch.cat([pred_flow[0], torch.zeros(1, pred_flow[0].shape[1], pred_flow[0].shape[2]).cuda()], dim=0)
        
        # save_img(pred_flow_img, "pred_flow.png")
        
        # visualie_flow(mini_batch['og_src_img'][0], mini_batch['og_trg_img'][0], pred_flow[0], "flow_est")
        # exit()
        
        # print("pred_flow.shape")
        # print(pred_flow.shape)
        # print("flow_gt.shape")
        # print(flow_gt.shape)
        # exit()
        
        # print("mini_batch['trg_kps']")
        # print(mini_batch['trg_kps'])

        # estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))
        
        # print("estimated_kps")
        # print(estimated_kps)
        # exit()
        
        # print("estimated_kps.shape")
        # print(estimated_kps.shape)
        # print("estimated_kps")
        # print(estimated_kps)
        # print("mini_batch['src_kps'].shape")
        # print(mini_batch['src_kps'].shape)
        # print("mini_batch['src_kps']")
        # print(mini_batch['src_kps'])
        # exit()


        eval_result = Evaluator.eval_kps_transfer(est_keypoints.cpu(), mini_batch)
        
        # Loss = EPE(pred_flow, flow_gt) 

        pck_array += eval_result['pck']

        # running_total_loss += Loss.item()
        # pbar.set_description(
        #     ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)
        
        
        print(f"{i} this pck ", eval_result['pck'], " mean_pck " , mean_pck)

    return running_total_loss / len(val_loader), mean_pck