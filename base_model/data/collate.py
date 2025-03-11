import torch
import random


def gen_mask(global_crops, n_tokens, mask_probability, mask_ratio_tuple, mask_generator):
    B = len(global_crops)  # bs*global_crops_num
    N = n_tokens    # 196
    n_samples_masked = int(B * mask_probability)    # bs*global_crops_num*mask_probability
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1) # tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000])  
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i] # 0.1/0.2/0.3/0.4
        prob_max = probs[i + 1] # 0.2/0.3/0.4/0.5
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))    # list of torch.Size([14, 14])
        upperbound += int(N * prob_max) # int(0.2*196)+int(0.3*196)+int(0.4*196)+int(0.5*196)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))    # list of torch.Size([14, 14])

    random.shuffle(masks_list)
    collated_masks = torch.stack(masks_list).flatten(1) # torch.Size([bs*global_crops_num, 196]) 
    mask_indices_list = collated_masks.flatten().nonzero().flatten()    # find out the indices of True 
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]  # same shape as mask_indices_list

    return collated_masks, upperbound, mask_indices_list, masks_weight



def collate_data_and_cast_with_aux_use_past_future_frames(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # print(len(samples_list))   # batch size
    # print(len(samples_list[0]))   # 2: [image_list, label]
    # print(len(samples_list[0][0]))   # 3 [past_frame, current_frame, future_frame]
    # print(samples_list[0][1])   # label
    # print(samples_list[0][0][0].keys(), samples_list[0][0][1].keys(), samples_list[0][0][2].keys())   # past_frame_dict, current_frame_dict, future_frame_dict
    
    n_global_crops = len(samples_list[0][0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0][0]["local_crops"])

    collated_global_crops_current = torch.stack([s[0][1]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops_current = torch.stack([s[0][1]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    
    collated_global_crops_past = torch.stack([s[0][0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_global_crops_future = torch.stack([s[0][2]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_masks, upperbound, mask_indices_list, masks_weight = gen_mask(collated_global_crops_current, n_tokens, mask_probability, mask_ratio_tuple, mask_generator)

    return {
        "collated_global_crops": collated_global_crops_current.to(dtype),
        "collated_local_crops": collated_local_crops_current.to(dtype),
        "collated_global_crops_past": collated_global_crops_past.to(dtype),
        "collated_global_crops_future": collated_global_crops_future.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long)
    }

