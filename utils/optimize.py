from tqdm import tqdm
import torch
from utils.utils import visualie_correspondences
from utils.evaluation import Evaluator

from utils.optimize_token import optimize_prompt, find_max_pixel_value, visualize_image_with_points, run_image_with_tokens_cropped

import wandb

def validate_epoch(ldm,
                   val_loader,
                   upsample_res = 512,
                   num_steps=100,
                   noise_level = 10,
                   layers = [0, 1, 2, 3, 4, 5],
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
                   num_iterations=20):
    

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    pck_array = []
    pck_array_ind_layers = [[] for i in range(len(layers))]
    for i, mini_batch in pbar:
        
        est_keypoints = -1*torch.ones_like(mini_batch['src_kps'])
        ind_layers = -1*torch.ones_like(mini_batch['src_kps']).repeat(len(layers), 1, 1)
            
        all_contexts = []

        for j in range(mini_batch['src_kps'].shape[2]):
            
            if mini_batch['src_kps'][0, 0, j] == -1:
                break
            
            if visualize:
                visualize_image_with_points(mini_batch['src_img'][0], mini_batch['src_kps'][0, :, j], f"{i:03d}_initial_point_{j:02d}", save_folder=save_folder)
                visualize_image_with_points(mini_batch['trg_img'][0], mini_batch['trg_kps'][0, :, j], f"{i:03d}_target_point_{j:02d}", save_folder=save_folder)
        
            # Find the text embeddings for the source point
            contexts = []
            for _ in range(num_opt_iterations):
                context = optimize_prompt(ldm, mini_batch['src_img'][0], mini_batch['src_kps'][0, :, j]/512, num_steps=num_steps, device=device, layers=layers, lr = lr, upsample_res=upsample_res, noise_level=noise_level, sigma = sigma, flip_prob=flip_prob, crop_percent=crop_percent)
                contexts.append(context)
            all_contexts.append(torch.stack(contexts))
            
            # Find and combine the attention maps over the multiple found text embeddings and crops
            all_maps = []
            for context in contexts:
                maps = []
                attn_maps, _ = run_image_with_tokens_cropped(ldm, mini_batch['trg_img'][0], context, index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations=num_iterations, image_mask = None if 'bool_img_trg' not in mini_batch else mini_batch['bool_img_trg'][0])
                for k in range(attn_maps.shape[0]):
                    avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                    maps.append(avg)
                    _max_val = find_max_pixel_value(avg[0], img_size = 512)
                    ind_layers[k, :, j] = (_max_val+0.5)
                maps = torch.stack(maps, dim=0)
                all_maps.append(maps)
            all_maps = torch.stack(all_maps, dim=0)
            all_maps = torch.mean(all_maps, dim=0)
            all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res*upsample_res))
            all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
            
            # Visualize the attention maps for the target image
            if visualize:
                for k in range(all_maps.shape[0]):
                    visualize_image_with_points(all_maps[k, None], mini_batch['trg_kps'][0, :, j]/512*upsample_res, f"{i:03d}_largest_loc_trg_{j:02d}_{k:02d}", save_folder=save_folder)
                visualize_image_with_points(torch.mean(all_maps, dim=0)[None], None, f"{i:03d}_largest_loc_trg_{j:02d}_mean", save_folder=save_folder)
            
            # Take the argmax to find the corresponding location for the target image
            all_maps = torch.mean(all_maps, dim=0)
            max_val = find_max_pixel_value(all_maps, img_size = 512)
            est_keypoints[0, :, j] = (max_val+0.5)
            
            
            # Find the attention maps for the source image
            if visualize:
                all_maps = []
                for context in contexts:
                    attn_map_src, _ = run_image_with_tokens_cropped(ldm, mini_batch['src_img'][0], context, index=0, upsample_res=upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations=num_iterations, image_mask = None if 'bool_img_src' not in mini_batch else mini_batch['bool_img_src'][0])
                    maps = []
                    for k in range(attn_map_src.shape[0]):
                        avg = torch.mean(attn_map_src[k], dim=0, keepdim=True)
                        maps.append(avg)
                    maps = torch.stack(maps, dim=0)
                    all_maps.append(maps)
                all_maps = torch.stack(all_maps, dim=0)
                all_maps = torch.mean(all_maps, dim=0)
                all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), upsample_res*upsample_res))
                all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
                for k in range(all_maps.shape[0]):  
                    visualize_image_with_points(all_maps[k, None], mini_batch['src_kps'][0, :, j]/512*upsample_res, f"{i:03d}_largest_loc_src_{j:02d}_{k:02d}", save_folder=save_folder)
                visualize_image_with_points(torch.mean(all_maps, dim=0)[None], None, f"{i:03d}_largest_loc_src_{j:02d}_mean", save_folder=save_folder)

        # Evaluate the performance of the individual layers
        for k in range(len(pck_array_ind_layers)):
            _eval_result = Evaluator.eval_kps_transfer(ind_layers[k].cpu()[None], mini_batch)
            pck_array_ind_layers[k] += _eval_result['pck']
            
            print(f"layer {k} pck {sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])}, this pck {_eval_result['pck']}")

        # Evaluate the performance of the estimated keypoints
        eval_result = Evaluator.eval_kps_transfer(est_keypoints.cpu(), mini_batch)
        
        # Visualize correspondence
        if visualize:
            visualie_correspondences(mini_batch['src_img'][0], mini_batch['trg_img'][0], mini_batch['src_kps'], est_keypoints, f"correspondences_estimated_{i:03d}", correct_ids = eval_result['correct_ids'], save_folder=save_folder)
            visualie_correspondences(mini_batch['src_img'][0], mini_batch['trg_img'][0], mini_batch['src_kps'], mini_batch['trg_kps'], f"correspondences_gt_{i:03d}", correct_ids = eval_result['correct_ids'], save_folder=save_folder)
            
        dict = {"est_keypoints": est_keypoints, "correct_ids": eval_result['correct_ids'], "src_kps": mini_batch['src_kps'], "trg_kps": mini_batch['trg_kps'], "idx": mini_batch['idx'] if item_index == -1 else item_index, "contexts": torch.stack(all_contexts), 'pck': eval_result['pck']}
        # save dict 
        torch.save(dict, f"{save_folder}/correspondence_data_{i:03d}.pt")

        pck_array += eval_result['pck']

        mean_pck_ten = sum(pck_array[1::2]) / len(pck_array[1::2])
        
        print(f"epoch: {epoch} {i} this pck ", eval_result['pck'], " mean_pck " , mean_pck_ten)
        
        if wandb_log:
            wandb_dict = {"pck": mean_pck_ten}
            for k in range(len(pck_array_ind_layers)):
                wandb_dict[f"pck_layer_{k}"] = sum(pck_array_ind_layers[k]) / len(pck_array_ind_layers[k])
            wandb.log(wandb_dict)

    return pck_array



def retest(ldm,
            test_dataset,
            upsample_res = 512,
            noise_level = 10,
            layers = [0, 1, 2, 3, 4, 5],
            epoch = 6,
            crop_percent=100.0,
            device = 'cpu',
            visualize = False,
            wandb_log = False,
            item_index = -1,
            ablate_results = False,
            num_iterations = 20,
            results_loc = "/scratch/iamerich/prompt-to-prompt/outputs/ldm_visualization_020",
            save_folder = "outputs"):
    """
    Takes the saved text embeddings and re-evaluates them
    
    if ablate_results:
        saves performance for average of 1-5 optimization iterations
        saves performance for average of 1-20 inference iterations
        saves performance of each layer
    """
    
    from glob import glob
    
    correspondences = glob(f"{results_loc}/*/correspondence_data_*.pt")
    
    if item_index != -1:
        correspondences = [correspondences[item_index]]
    
    index = 0

    pck_array = []
    pck_array_ind_layers = [[] for _ in range(len(layers))]
    # for i, correspondence in enumerate(correspondences):
    for _i in range(len(correspondences)):
        
        i = int(correspondences[_i].split("/")[-2])
        
        data = torch.load(correspondences[_i], map_location=device)
        
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
        
        
        for j in range(contexts.shape[0]):
            print(j)
            
            assert mini_batch['src_kps'][0, j] != -1
            
            if visualize:
                visualize_image_with_points(mini_batch['src_img'], mini_batch['src_kps'][:, j], f"{index:03d}_initial_point_{j:02d}", save_folder=save_folder)
                visualize_image_with_points(mini_batch['trg_img'], mini_batch['trg_kps'][0, :, j], f"{index:03d}_target_point_{j:02d}", save_folder=save_folder)
            
            all_maps = []
            
            collected_attention_maps = []
            
            # for context in contexts:
            for l in range(contexts.shape[1]):
                
                maps = []
        
                attn_maps, _collected_attention_maps = run_image_with_tokens_cropped(ldm, mini_batch['trg_img'], contexts[j, l].to(device), index=0, upsample_res = upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations = num_iterations)
                
                collected_attention_maps.append(torch.stack(_collected_attention_maps, dim=0).detach().cpu())
                
                for k in range(attn_maps.shape[0]):
                    avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                    maps.append(avg)
                    assert avg.shape[0] == 1
                    _max_val = find_max_pixel_value(avg[0], img_size = 512)
                    ind_layers[k, :, j] = (_max_val+0.5)

                maps = torch.stack(maps, dim=0)
                
                all_maps.append(maps)
                
                if ablate_results:
                    
                    # save performance per optimization iteration
                    mean_this_it = torch.mean(torch.stack(all_maps, dim=0), dim=0)
                    mean_this_it = torch.mean(mean_this_it, dim=0)
                    assert mean_this_it.shape[0] == 1
                    _max_val = find_max_pixel_value(mean_this_it[0], img_size = 512)
                    
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
            
            all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
            
            
            if visualize:
                for k in range(all_maps.shape[0]):
                    visualize_image_with_points(all_maps[k, None], mini_batch['trg_kps'][0, :, j]/512*upsample_res, f"{index:03d}_largest_loc_trg_{j:02d}_{k:02d}", save_folder=save_folder)
                visualize_image_with_points(torch.mean(all_maps, dim=0)[None], None, f"{index:03d}_largest_loc_trg_{j:02d}_mean", save_folder=save_folder)
                
                
            all_maps = torch.mean(all_maps, dim=0)
            max_val = find_max_pixel_value(all_maps, img_size = 512)
            est_keypoints[:, j] = (max_val+0.5)
            
            
            if visualize:
                
                all_maps = []
                
                for l in range(contexts.shape[1]):
                    
                    attn_map_src, _ = run_image_with_tokens_cropped(ldm, mini_batch['src_img'], contexts[j, l], index=0, upsample_res=upsample_res, noise_level=noise_level, layers=layers, device=device, crop_percent=crop_percent, num_iterations=num_iterations)
                    
                    maps = []
                    for k in range(attn_map_src.shape[0]):
                        avg = torch.mean(attn_map_src[k], dim=0, keepdim=True)
                        maps.append(avg)
                        
                    maps = torch.stack(maps, dim=0)
                
                    all_maps.append(maps)
                    
                all_maps = torch.stack(all_maps, dim=0)
                # take the average along dim=0
                all_maps = torch.mean(all_maps, dim=0)
            
                all_maps = all_maps.reshape(len(layers), upsample_res, upsample_res)
                        
                        
                for k in range(all_maps.shape[0]):  
                    visualize_image_with_points(all_maps[k, None], mini_batch['src_kps'][:, j]/512*upsample_res, f"{index:03d}_largest_loc_src_{j:02d}_{k:02d}", save_folder=save_folder)
                visualize_image_with_points(torch.mean(all_maps, dim=0)[None], None, f"{index:03d}_largest_loc_src_{j:02d}_mean", save_folder=save_folder)
                
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
            visualie_correspondences(mini_batch['src_img'], mini_batch['trg_img'], mini_batch['src_kps'][None], est_keypoints[None], f"correspondences_estimated_{index:03d}_{eval_result['pck'][1]}%", correct_ids = eval_result['correct_ids'], save_folder=save_folder, line_width=5)
            visualie_correspondences(mini_batch['src_img'], mini_batch['trg_img'], mini_batch['src_kps'][None], mini_batch['trg_kps'], f"correspondences_gt_{index:03d}_{eval_result['pck'][1]}%", correct_ids = eval_result['correct_ids'], save_folder=save_folder, line_width=5)

        pck_array += eval_result['pck']

        mean_pck = sum(pck_array) / len(pck_array)
        
        index += 1
        
        dict = {"est_keypoints": est_keypoints, "correct_ids": eval_result['correct_ids'], "src_kps": mini_batch['src_kps'], "trg_kps": mini_batch['trg_kps'], "idx": mini_batch['idx'] if item_index == -1 else item_index, 'pck': eval_result['pck']}
        # save dict 
        torch.save(dict, f"{save_folder}/correspondence_data_{index:03d}.pt")
        
        print(f"epoch: {epoch} {i} this pck ", eval_result['pck'], " mean_pck " , mean_pck)
        
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
