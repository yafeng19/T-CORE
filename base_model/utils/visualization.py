
import os
import torch
import matplotlib
import seaborn as sns
import numpy as np
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image



def save_weights(iteration, debug_save_path, 
                      frame_name_current, frame_name_past, frame_name_future,
                      past_cross_attn_weights, future_cross_attn_weights):
    data_to_save = {
        "attention_weights": past_cross_attn_weights,
        "frame_name_current": frame_name_current,
        "frame_name_past": frame_name_past
    }
    save_dir = os.path.join(debug_save_path, 'attn_weights', f'iter_{iteration}_current_past.pt')
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    torch.save(data_to_save, save_dir)

    data_to_save = {
        "attention_weights": future_cross_attn_weights,
        "frame_name_current": frame_name_current,
        "frame_name_future": frame_name_future
    }
    save_dir = os.path.join(debug_save_path, 'attn_weights', f'iter_{iteration}_current_future.pt')
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    torch.save(data_to_save, save_dir)


def load_weights(iteration, debug_save_path):

    weights_dir = os.path.join(debug_save_path, 'attn_weights', f'iter_{iteration}_current_past.pt')
    past_attn_dict = torch.load(weights_dir)
    # print(past_attn_dict["attention_weights"].shape)
    # print(past_attn_dict["frame_name_current"])
    # print(past_attn_dict["frame_name_past"])

    weights_dir = os.path.join(debug_save_path, 'attn_weights', f'iter_{iteration}_current_future.pt')
    future_attn_dict = torch.load(weights_dir)
    # print(future_attn_dict["attention_weights"].shape)
    # print(future_attn_dict["frame_name_current"])
    # print(future_attn_dict["frame_name_future"])

    return past_attn_dict, future_attn_dict


def visualize_weights(iteration, debug_save_path, past_attn_dict, future_attn_dict):
    attention_weights = past_attn_dict["attention_weights"].numpy()
    frame_name_current = past_attn_dict["frame_name_current"]
    frame_name_past = past_attn_dict["frame_name_past"]
    bs = len(frame_name_current)
    
    for idx in range(bs):
        plt.figure(figsize=(8, 6))
        sns.heatmap(attention_weights[idx], cmap='viridis', cbar=True)
        # print(f"Current Frame: {frame_name_current[idx]}\nPast Frame: {frame_name_past[idx]}")
        plt.xlabel(f"Current Frame: {frame_name_current[idx]}\nPast Frame: {frame_name_past[idx]}")
        fig_path = os.path.join(debug_save_path, 'attn_weight_maps', f'iter_{iteration}')
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, 'past_attn_weight_maps_{:04d}.png'.format(idx+1)), bbox_inches='tight', pad_inches=0)
        plt.clf()

    attention_weights = future_attn_dict["attention_weights"].numpy()
    frame_name_current = future_attn_dict["frame_name_current"]
    frame_name_future = future_attn_dict["frame_name_future"]
    bs = len(frame_name_current)
    
    for idx in range(bs):
        plt.figure(figsize=(8, 6))
        sns.heatmap(attention_weights[idx], cmap='viridis', cbar=True)
        plt.xlabel(f"Current Frame: {frame_name_current[idx]}\nFuture Frame: {frame_name_future[idx]}")
        fig_path = os.path.join(debug_save_path, 'attn_weight_maps', f'iter_{iteration}')
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, 'future_attn_weight_maps_{:04d}.png'.format(idx+1)), bbox_inches='tight', pad_inches=0)
        plt.clf()



def save_and_visualize_attn_maps(iteration, debug_save_path, 
                      frame_name_current, frame_name_past, frame_name_future,
                      past_cross_attn_weights, future_cross_attn_weights):
    
    save_weights(iteration, debug_save_path, 
                      frame_name_current, frame_name_past, frame_name_future,
                      past_cross_attn_weights, future_cross_attn_weights)
    
    # past_attn_dict, future_attn_dict = load_weights(iteration, debug_save_path)

    # visualize_weights(iteration, debug_save_path, past_attn_dict, future_attn_dict)




def save_masks(iteration, debug_save_path, frame_name_current, current_crops, current_masks):
    data_to_save = {
        "frame_name_current": frame_name_current,
        "current_crops": current_crops,
        "current_masks": current_masks,
    }
    save_dir = os.path.join(debug_save_path, 'current_masks', f'iter_{iteration}_current_masks.pt')
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    torch.save(data_to_save, save_dir)


def load_masks(iteration, debug_save_path):
    masks_dir = os.path.join(debug_save_path, 'current_masks', f'iter_{iteration}_current_masks.pt')
    masks_dict = torch.load(masks_dir)
    # print(masks_dict["curren_crops"].shape)
    # print(masks_dict["curren_masks"].shape)

    return masks_dict



def save_and_visualize_masks(iteration, debug_save_path, frame_name_current, current_crops, current_masks):
    
    save_masks(iteration, debug_save_path, frame_name_current, current_crops, current_masks)
    
    # masks_dict = load_masks(iteration, debug_save_path)





def visualize_grid_attention(query_img_path, refer_img_path, save_path, query_patch_pos_tuple, attention_mask, ratio=1, cmap="jet", save_image=False,
                             save_original_image=False, save_query_image=True, quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    # print("load image from: ", img_path)
    img = Image.open(refer_img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.4, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = refer_img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)
        
        # pre-process and save image
        # print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)
        plt.close()

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        # print("save original image at the same time")
        img_name = refer_img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)

    plt.clf()

    q_img = Image.open(query_img_path, mode='r')
    q_img_h, q_img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * q_img_h, 0.02 * q_img_w))

    # scale the image
    q_img_h, q_img_w = int(q_img.size[0] * ratio), int(q_img.size[1] * ratio)
    q_img = q_img.resize((q_img_h, q_img_w))
    plt.imshow(q_img, alpha=1)
    plt.axis('off')

    x, y = query_patch_pos_tuple
    patch_w = q_img_w / 14
    patch_h = q_img_h / 14
    rect_x = x * patch_w
    rect_y = y * patch_h

    rect = patches.Rectangle((rect_x, rect_y), patch_w, patch_h, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

    if save_query_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = query_img_path.split('/')[-1].split('.')[0] + "_with_query.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)
        
        # pre-process and save image
        # print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)
        plt.close()
