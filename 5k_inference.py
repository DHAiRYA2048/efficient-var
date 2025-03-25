import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt
# Removed safety_check import

def save_image(image_tensor, file_path):
    """Save a single image tensor to a file.
    
    The function handles different tensor shapes:
      - (3, H, W): assumed to be RGB and transposed to (H, W, 3)
      - (H, W, 3): assumed to be RGB already
      - (H, W): grayscale image
    """
    # Squeeze out singleton batch dimension if present.
    if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)

    # Convert tensor from [0,1] to [0,255] and to numpy
    image_np = image_tensor.mul(255).cpu().numpy().astype(np.uint8)

    # Handle tensor shape conversion based on number of dimensions
    if image_np.ndim == 3:
        if image_np.shape[0] == 3:
            # Tensor shape is (3, H, W) -> transpose to (H, W, 3)
            image_np = image_np.transpose(1, 2, 0)
        elif image_np.shape[-1] == 3:
            # Already (H, W, 3), do nothing
            pass
        else:
            raise ValueError(f"Unexpected 3D image shape: {image_np.shape}")
    elif image_np.ndim == 2:
        # Grayscale image, no need to transpose.
        pass
    else:
        raise ValueError(f"Unexpected image tensor shape: {image_np.shape}")

    im = Image.fromarray(image_np)
    im.save(file_path)
    print(f"Saved image {file_path}")

def main(args):
    device = torch.device("cuda")

    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    # If a JSONL file is provided, load pairs and generate images for each prompt_a and prompt_b.
    if args.jsonl_file:
        with open(args.jsonl_file, "r") as f:
            pairs = [json.loads(line) for line in f]
        print(f"Loaded {len(pairs)} pairs from {args.jsonl_file}")

        # Loop over each pair (starting indexing at 1)
        for idx, pair in enumerate(pairs, start=1):
            for variant, key in zip(["a", "b"], ["prompt_a", "prompt_b"]):
                prompt = pair[key]
                

                with torch.inference_mode():
                    with torch.autocast(
                        "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
                    ):
                        # Encode the prompt (as a list with one element)
                        (
                            context_tokens,
                            context_mask,
                            context_position_ids,
                            context_tensor,
                        ) = encode_prompts(
                            [prompt],
                            text_model,
                            text_tokenizer,
                            args.max_token_length,
                            llm_system_prompt,
                            args.use_llm_system_prompt,
                        )

                        infer_func = (
                            ema_model.autoregressive_infer_cfg
                            if args.use_ema
                            else model.autoregressive_infer_cfg
                        )
                        output_imgs = infer_func(
                            B=context_tensor.size(0),
                            label_B=context_tensor,
                            cfg=args.cfg,
                            g_seed=args.seed,
                            more_smooth=args.more_smooth,
                            context_position_ids=context_position_ids,
                            context_mask=context_mask,
                        )
                # Save the generated image with the naming format "<entry>_<variant>.png"
                os.makedirs(args.sample_folder_dir, exist_ok=True)
                image_filename = f"{idx}_{variant}.png"
                image_path = os.path.join(args.sample_folder_dir, image_filename)
                # output_imgs is a batch with one image so we index [0]
                save_image(output_imgs[0].clone(), image_path)
    else:
        # Fallback behavior: use prompt or prompt_list or default prompts.
        prompts = []
        if args.prompt:
            prompts = [args.prompt]
        elif args.prompt_list:
            prompts = args.prompts
        else:
            print(
                "No prompt is provided. Will randomly sample 2 prompts from default prompts."
            )
            prompts = random.sample(default_prompts, 2)

        start_time = time.time()
        with torch.inference_mode():
            with torch.autocast(
                "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
            ):
                (
                    context_tokens,
                    context_mask,
                    context_position_ids,
                    context_tensor,
                ) = encode_prompts(
                    prompts,
                    text_model,
                    text_tokenizer,
                    args.max_token_length,
                    llm_system_prompt,
                    args.use_llm_system_prompt,
                )

                infer_func = (
                    ema_model.autoregressive_infer_cfg
                    if args.use_ema
                    else model.autoregressive_infer_cfg
                )
                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )

        total_time = time.time() - start_time
        print(f"Generate {len(prompts)} images took {total_time:2f}s.")

        # Save images separately with index-based naming.
        os.makedirs(args.sample_folder_dir, exist_ok=True)
        for img_idx in range(output_imgs.size(0)):
            image_filename = f"{img_idx+1:06d}.png"
            image_path = os.path.join(args.sample_folder_dir, image_filename)
            save_image(output_imgs[img_idx].clone(), image_path)

        with open(os.path.join(args.sample_folder_dir, "prompt.txt"), "w") as f:
            f.write("\n".join(prompts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="/scratch/dps9998/hart_inference/hart-0.7b-1024px/llm/",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="/scratch/dps9998/hart_inference/Qwen2-VL-1.5B-Instruct/",
    )
    parser.add_argument("--prompt", type=str, help="A single prompt.", default="")
    parser.add_argument("--prompt_list", type=list, default=[])
    # New argument to provide the JSONL file with prompt pairs.
    parser.add_argument("--jsonl_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    # Update default folder to "hart_fid_baseline"
    parser.add_argument(
        "--sample_folder_dir",
        type=str,
        help="The folder where the image samples are stored",
        default="/scratch/dps9998/hart_inference/hart_fid_baseline",
    )
    parser.add_argument(
        "--store_seperately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)

