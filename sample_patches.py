import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt

def save_images(final_imgs, intermediate_imgs, final_folder_dir, intermediate_folder_dir, store_seperately, prompts):
    # Create output directories if they don't exist.
    os.makedirs(final_folder_dir, exist_ok=True)
    os.makedirs(intermediate_folder_dir, exist_ok=True)
    
    # --- Save final images ---
    if not store_seperately and final_imgs.size(0) > 1:
        # Create a grid from the final images
        grid = torchvision.utils.make_grid(final_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
        final_image = Image.fromarray(grid_np)
        final_path = os.path.join(final_folder_dir, "final_sample_images.png")
        final_image.save(final_path)
        print(f"Final image grid saved to {final_path}")
    else:
        # Save each image separately
        sample_imgs_np = final_imgs.mul(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            img_path = os.path.join(final_folder_dir, f"{img_idx:06d}.png")
            cur_img_store.save(img_path)
            print(f"Final image {img_idx} saved to {img_path}")
    
    # --- Save intermediate scales ---
    num_scales = len(intermediate_imgs)
    if num_scales > 0:
        # Create a subplot for each scale
        fig, axes = plt.subplots(1, num_scales, figsize=(num_scales * 2, 2))
        # If there is only one scale, axes might not be a list
        if num_scales == 1:
            axes = [axes]
        for i, inter_img in enumerate(intermediate_imgs):
            # Create a grid from the intermediate images at this scale
            inter_img_grid = torchvision.utils.make_grid(inter_img, nrow=4, padding=0, pad_value=1.0)
            inter_img_grid = inter_img_grid.clone().permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
            axes[i].imshow(inter_img_grid)
            axes[i].axis("off")
            height, width, _ = inter_img_grid.shape
            axes[i].set_title(f"{width}X{height}")
        plt.tight_layout()
        intermediate_path = os.path.join(intermediate_folder_dir, "intermediate_scales.png")
        plt.savefig(intermediate_path)
        plt.close(fig)
        print(f"Intermediate scales figure saved to {intermediate_path}")
    
    # --- Save prompts ---
    with open(os.path.join(final_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))
    print("Prompts saved.")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the HART model
    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"), map_location=device)
        )
    else:
        ema_model = None
    
    # Load the text model and tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()
    
    # Determine prompt(s)
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_list:
        prompts = args.prompt_list
    else:
        print("No prompt provided. Sampling default prompts.")
        prompts = random.sample(default_prompts, 1)
    
    # Encode the prompt(s)
    context_tokens, context_mask, context_position_ids, context_tensor = encode_prompts(
        prompts,
        text_model,
        text_tokenizer,
        args.max_token_length,
        llm_system_prompt,
        args.use_llm_system_prompt,
    )
    
    # Run inference and get both final and intermediate images
    start_time = time.time()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            infer_func = ema_model.autoregressive_infer_cfg if args.use_ema else model.autoregressive_infer_cfg
            final_imgs, intermediate_imgs = infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=args.cfg,
                g_seed=args.seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )
    total_time = time.time() - start_time
    print(f"Generated {len(prompts)} image(s) in {total_time:.2f}s.")
    
    # Save both final and intermediate images to disk
    save_images(final_imgs.clone(), intermediate_imgs, args.final_folder_dir, args.intermediate_folder_dir, args.store_seperately, prompts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/scratch/dps9998/hart_inference/hart-0.7b-1024px/llm/", help="Path to HART model")
    parser.add_argument("--text_model_path", type=str, default="/scratch/dps9998/hart_inference/Qwen2-VL-1.5B-Instruct/", help="Path to text model")
    parser.add_argument("--prompt", type=str, default="", help="A single prompt")
    parser.add_argument("--prompt_list", type=str, nargs='+', default=[], help="List of prompts")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--more_smooth", type=bool, default=True)
    parser.add_argument("--store_seperately", action="store_true", help="Store images separately")
    parser.add_argument("--final_folder_dir", type=str, default="final_outputs", help="Directory to store final images")
    parser.add_argument("--intermediate_folder_dir", type=str, default="intermediate_scales", help="Directory to store intermediate scale images")
    args = parser.parse_args()
    
    main(args)

