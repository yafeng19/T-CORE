
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from base_model.utils.visualization import visualize_grid_attention
import numpy as np



def vis_attn_map(prefix_path, iter_id):
    # past
    past_pth = 'iter_' + str(iter_id) + '_current_past.pt'
    weights_dir = os.path.join('debug_save', 'attn_weights', past_pth)
    past_attn_dict = torch.load(weights_dir)
    attention_weights = past_attn_dict["attention_weights"]
    frame_name_current = past_attn_dict["frame_name_current"]
    frame_name_past = past_attn_dict["frame_name_past"]
    bs, n_patch, _ = attention_weights.shape
    for idx in range(bs):
        plt.figure(figsize=(8, 6))
        for row in range(n_patch):
            w, h = row//14, row%14
            current_append_path = os.path.join(prefix_path, frame_name_current[idx].replace('#', '/') + '.jpg')
            past_append_path = os.path.join(prefix_path, frame_name_past[idx].replace('#', '/') + '.jpg')
            fig_path = os.path.join('debug_save', 'attn_maps', 'iter_'+str(iter_id), 'attn_maps_{:02d}_({:02d},{:02d})'.format(idx+1, h, w))
            os.makedirs(fig_path, exist_ok=True)
            attn = attention_weights[idx][row].view(14, 14).float().numpy()
            # print(attn)
            visualize_grid_attention(
                query_img_path=current_append_path, 
                refer_img_path=past_append_path, 
                save_path=fig_path,
                query_patch_pos_tuple=(h, w),
                attention_mask=attn,
                save_image=True,
                save_original_image=True,
                save_query_image=True,
                quality=100)

    # future
    future_pth = 'iter_' + str(iter_id) + '_current_future.pt'
    weights_dir = os.path.join('debug_save', 'attn_weights', future_pth)
    future_attn_dict = torch.load(weights_dir)
    attention_weights = future_attn_dict["attention_weights"]
    frame_name_current = future_attn_dict["frame_name_current"]
    frame_name_future = future_attn_dict["frame_name_future"]
    bs, n_patch, _ = attention_weights.shape
    for idx in range(bs):
        plt.figure(figsize=(8, 6))
        for row in range(n_patch):
            w, h = row//14, row%14
            current_append_path = os.path.join(prefix_path, frame_name_current[idx].replace('#', '/') + '.jpg')
            future_append_path = os.path.join(prefix_path, frame_name_future[idx].replace('#', '/') + '.jpg')
            fig_path = os.path.join('debug_save', 'attn_maps', 'iter_'+str(iter_id), 'attn_maps_{:02d}_({:02d},{:02d})'.format(idx+1, h, w))
            os.makedirs(fig_path, exist_ok=True)
            attn = attention_weights[idx][row].view(14, 14).float().numpy()
            # print(attn)
            visualize_grid_attention(
                query_img_path=current_append_path, 
                refer_img_path=future_append_path, 
                save_path=fig_path,
                query_patch_pos_tuple=(h, w),
                attention_mask=attn,
                save_image=True,
                save_original_image=True,
                save_query_image=False,
                quality=100)



def vis_mask(save_path, iter_id):
    mask_pth = 'iter_' + str(iter_id) + '_current_masks.pt'
    weights_dir = os.path.join('debug_save', 'current_masks', mask_pth)
    mask_dict = torch.load(weights_dir)
    # current_crops = mask_dict["current_crops"]
    frame_name_current = mask_dict["frame_name_current"]
    current_masks = mask_dict["current_masks"]
    # print(current_crops.shape)  # torch.Size([bs*2, 3, 224, 224])
    # print(current_masks.shape)  # torch.Size([bs*2, 196])
    bs, n_patch = current_masks.shape
    for idx in range(bs):
        current_mask = current_masks[idx].view(14, 14)
        current_mask = current_mask.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)
        # current_mask = current_mask.unsqueeze(-1).expand(-1, -1, 3)
        current_mask_np = current_mask.numpy().astype(bool)

        # print(current_mask.shape)    # torch.Size([224, 224, 3])
        # current_crop = current_crops[idx]
        current_append_path = os.path.join(prefix_path, frame_name_current[idx].replace('#', '/') + '.jpg')
        
        image = Image.open(current_append_path).resize((224, 224))
        image_np = np.array(image.convert("RGB"))
        # print(image_np.shape) (224, 224, 3)

        masked_image = image_np.copy()
        masked_image[current_mask_np] = [123, 116, 103]

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = f"iter_{iter_id}_current_{idx+1}_with_mask.jpg"
        img_with_mask_save_path = os.path.join(save_path, img_name)
        
        plt.figure(figsize=(224*0.02, 224*0.02))
        plt.axis('off')
        plt.imshow(masked_image)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_mask_save_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()

            

if __name__ == '__main__':
    prefix_path = "debug_save/crops"
    mask_save_path = "debug_save"
    start_iter = 374401 # well-trained model
    end_iter = 374401
    for iter_id in range(start_iter, end_iter+1):
        # vis_mask(mask_save_path, iter_id)
        vis_attn_map(prefix_path, iter_id)
        

