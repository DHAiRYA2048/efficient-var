import argparse
import copy
import datetime
import os
import random
import time
from typing import Optional, Union
import math
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

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).cuda(),
    ).bool()
    return masking

def sample_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples=1,
) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (
            1 - top_p
        )
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)


def gumbel_softmax_with_rng(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    rng: torch.Generator = None,
) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_(generator=rng)
        .log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def save_images(sample_imgs, sample_folder_dir, store_seperately, prompts):
    os.makedirs(sample_folder_dir, exist_ok=True)
    # Save each image individually as "0.png", "1.png", etc.
    sample_imgs_np = sample_imgs.mul(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    for img_idx in range(num_imgs):
        cur_img = sample_imgs_np[img_idx].transpose(1, 2, 0).astype(np.uint8)
        im = Image.fromarray(cur_img)
        im.save(os.path.join(sample_folder_dir, f"{img_idx}.png"))
        print(f"Image {img_idx} saved.")
    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


def copy_rng_state(rng_state):
    if isinstance(rng_state, tuple):
        return tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in rng_state)
    elif isinstance(rng_state, torch.Tensor):
        return rng_state.clone()
    else:
        return rng_state


def clone_cache_state(cache_state):
    cloned = []
    for (k, v) in cache_state:
        k_clone = k.detach().clone() if k is not None else None
        v_clone = v.detach().clone() if v is not None else None
        cloned.append((k_clone, v_clone))
    return cloned

def generate_initial_patch_with_context(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        label_B2,
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
    ):
        """
        Generate the initial patch with context (first stage only).
        Returns state variables needed to continue generation in the next function.
        
        Returns:
            f_hat: Latent state tensor
            next_token_map: Updated token map for stage 1
            rng_state: RNG state after initial stage
            cur_L: Cumulative token count
            cond_BD: The condition tensor used
            cache_state: KV cache state
            context_variables: Dictionary with context-related variables
        """
        # Set up RNG for deterministic behavior if requested
        if g_seed is not None:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        else:
            rng = None
            
        assert label_B is not None
        assert label_B.shape[1] == self.context_token

        # Compute initial conditioning
        sos = cond_BD = self.context_embed(
            self.context_norm(
                torch.cat((label_B, torch.full_like(label_B, fill_value=0.0)), dim=0)
            )
        )

        cond_BD2 = self.context_embed(
            self.context_norm(
                torch.cat((label_B2, torch.full_like(label_B, fill_value=0.0)), dim=0)
            )
        )

        
        # Handle CFG by replicating context position ids
        context_position_ids_full = torch.cat(
            (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
            dim=0,
        )

        # Prepare context mask
        b = context_mask.shape[0]
        context_mask_full = torch.cat(
            (context_mask, torch.full_like(context_mask, fill_value=0))
        )
        context_mask_full[b:, 0] = 1

        # Compute positional embeddings
        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L)

        # Create token map for first stage
        if self.pos_start is not None:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )
        else:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1) + lvl_pos[:, : self.first_l]
            )

        # Initialize latent representation
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        # Enable KV caching for all blocks
        for b in self.blocks:
            b.attn.kv_caching(True)
            
        # Process first stage (si = 0)
        si = 0
        pn = self.patch_nums[0]
        ratio = si / self.num_stages_minus_1
        
        # Update token counter
        cur_L += self.context_token  # First stage includes context tokens
        
        # Process through transformer blocks
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in self.blocks:
            x = b(
                x=x,
                cond_BD=cond_BD_or_gss,
                attn_bias=None,
                si=si,
                context_position_ids=context_position_ids_full,
                context_mask=context_mask_full,
            )
        
        # Get logits and apply CFG
        logits_BlV = self.get_logits(x, cond_BD)
        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
        
        # Special handling for text-conditioned generation
        logits_BlV = logits_BlV[:, [-1], :]
        
        # Sample tokens
        idx_Bl = sample_with_top_k_top_p_(
            logits_BlV,
            rng=rng,
            top_k=(600 if si < 7 else 300),
            top_p=top_p,
            num_samples=1,
        )[:, :, 0]
        
        # Convert to embeddings
        if not more_smooth:
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
        else:
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = gumbel_softmax_with_rng(
                logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
            ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
        
        # Reshape and update latent representation
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
        f_hat, latent_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
        )
        
        # Prepare token map for next stage
        latent_token_map = latent_token_map.view(B, self.Cvae, -1).transpose(1, 2)

        token_count = self.patch_nums[si + 1] ** 2
        second_cond = cond_BD2[B:, :token_count, :]

        next_token_map = (
            self.word_embed(latent_token_map)
            + second_cond
            + lvl_pos[:, cur_L : cur_L + token_count]
        )

        next_token_map = next_token_map.repeat(2, 1, 1)        
        
        rng_state = self.rng.get_state()

        # Save KV cache state for each block
        cache_state = []
        for b in self.blocks:
            k = b.attn.cached_k.detach().clone() if b.attn.cached_k is not None else None
            v = b.attn.cached_v.detach().clone() if b.attn.cached_v is not None else None
            cache_state.append((k, v))
            
        # Package context variables
        context_variables = {
            "context_position_ids": context_position_ids_full,
            "context_mask": context_mask_full,
            "lvl_pos": lvl_pos
        }
            
        for b in self.blocks:
            b.attn.kv_caching(False)

        return f_hat, next_token_map, rng_state, cur_L, cond_BD, cache_state


def continue_autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        final_stage=0,
        num_maskgit_iters=1,
        init_f_hat=None,
        init_next_token_map=None,
        init_rng_state=None,
        init_cur_L=None,
        init_cond_BD=None,
        init_cache_state=None,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Continue autoregressive generation from the second stage and handle MaskGIT final stage.
        
        Returns:
            Reconstructed image (B, 3, H, W) in [0, 1]
        """
        
        for b in self.blocks:
            b.attn.kv_caching(False)

        if g_seed is not None:
            self.rng.manual_seed(g_seed)
        elif init_rng_state is not None:
            # Make sure the RNG state is properly set
            self.rng.set_state(init_rng_state)
        rng = self.rng

        sos = cond_BD = self.context_embed(
            self.context_norm(
                torch.cat((label_B, torch.full_like(label_B, fill_value=0.0)), dim=0)
            )
        )

        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L)


        if init_f_hat is None or init_next_token_map is None or init_cur_L is None:
            next_token_map = (
                cond_BD.unsqueeze(1).expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, :self.first_l]
            )
            f_hat = cond_BD.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
            cur_L = 0
        else:
            f_hat = init_f_hat
            next_token_map = init_next_token_map  # Already updated for stage 1.
            cur_L = init_cur_L  # Should equal patch_nums[0]^2 (e.g. 1)
    
        
        # Get context variables
        context_position_ids = torch.cat(
            (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
            dim=0,
        )

        b = context_mask.shape[0]
        context_mask = torch.cat(
            (context_mask, torch.full_like(context_mask, fill_value=0))
        )
        context_mask[b:, 0] = 1

             
               
        for b in self.blocks:
            b.attn.kv_caching(True)

        # Restore cache state if provided
        if init_cache_state is not None:
            for b, (cached_k, cached_v) in zip(self.blocks, init_cache_state):
                b.attn.cached_k = cached_k.detach().clone() if cached_k is not None else None
                b.attn.cached_v = cached_v.detach().clone() if cached_v is not None else None

        # Process remaining autoregressive stages (starting from second stage)
        for si, pn in enumerate(self.patch_nums[1:-1], start=1):  # Start from 1, skip last one for MaskGIT

            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            if si == self.num_stages_minus_1:
                last_layer_cond = x

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            
            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
            )

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(2, 1, 1)

        ################ last stage maskgit ################
        si = len(self.patch_nums) - 1
        mask = torch.ones(B, self.last_level_pns).cuda()
        tokens = torch.zeros(B, self.last_level_pns, self.Cvae).cuda()
        orders = self.sample_orders(B)

        num_iter = num_maskgit_iters
        indices = list(range(num_iter))
        # generate latents with maskgit
        for step in indices:
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.last_level_pns * mask_ratio)]).cuda()
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            mask_next = mask_by_order(mask_len[0], orders, B, self.last_level_pns)
            if step >= num_iter - 1:
                mask_to_pred = mask[:B].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:B].bool(), mask_next.bool())
            mask = mask_next
            cur_mask = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            cur_mask = cur_mask.nonzero(as_tuple=True)
            x = next_token_map[cur_mask].reshape(2 * B, -1, self.C)
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=len(self.patch_nums) - 1,
                    m_maskgit=cur_mask,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            last_layer_cond = x
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            si = len(self.patch_nums) - 1
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            if final_stage == 0:
                # sample with diffusion model
                last_stage_discrete_cond = self.vae_quant_proxy[0].embedding(idx_Bl)
                last_stage_discrete_cond = self.word_embed(last_stage_discrete_cond)
                last_stage_discrete_cond = torch.cat(
                    [last_stage_discrete_cond, last_stage_discrete_cond], dim=0
                )
                last_stage_cond = self.decoder_norm(
                    last_layer_cond + last_stage_discrete_cond
                )
                bs, cur_seq_len, _ = last_stage_cond.shape
                ##### begin baseline sampling #####
                last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
                h_BChw_diff = self.diffloss.sample(
                    z=last_stage_cond, temperature=1.0, cfg=t
                )
                ##### end baseline sampling #####
                h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
                # [B, L, Cvae]
                h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
                # update feature map
                tokens[mask_to_pred] = (h_BChw + h_BChw_diff).reshape(-1, self.Cvae)
            else:
                tokens[mask_to_pred] = h_BChw.reshape(-1, self.Cvae)
        h_BChw_final = tokens.transpose(1, 2).reshape(
            B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]
        )
        f_hat += h_BChw_final

        ################ last stage maskgit ################

        # Disable KV caching when done
        for b in self.blocks:
            b.attn.kv_caching(False)
            
        # Return normalized image
        return (
            self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        )  # de-normalize, from [-1, 1] to [0, 1]

def main(args):
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(torch.load(os.path.join(args.model_path, "ema_model.bin")))
    else:
        ema_model = None

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()

    # Get single prompt from arguments
    summary = args.summary
    prompt = args.prompt

    # Encode the prompt
    tokens1, mask1, position_ids1, tensor1 = encode_prompts(
        [summary],
        text_model,
        text_tokenizer,
        args.max_token_length,
        llm_system_prompt,
        args.use_llm_system_prompt,
    )

    tokens2, mask2, position_ids2, tensor2 = encode_prompts(
        [prompt],
        text_model,
        text_tokenizer,
        args.max_token_length,
        llm_system_prompt,
        args.use_llm_system_prompt,
    )


    # Prepare for inference
    infer_model = ema_model if args.use_ema else model
    
    # Add the split functions to the model if not already there
    if not hasattr(infer_model, "generate_initial_patch_with_context"):
        infer_model.generate_initial_patch_with_context = generate_initial_patch_with_context.__get__(infer_model)
    
    if not hasattr(infer_model, "continue_autoregressive_infer_cfg"):
        infer_model.continue_autoregressive_infer_cfg = continue_autoregressive_infer_cfg.__get__(infer_model)

    # Create output directory if it doesn't exist
    os.makedirs(args.sample_folder_dir, exist_ok=True)

    def copy_rng_state(rng_state):
        # Properly copy RNG state
        # The full state may contain multiple tensors
        if isinstance(rng_state, tuple):
            return tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in rng_state)
        elif isinstance(rng_state, torch.Tensor):
            return rng_state.clone()
        else:
            return rng_state  # For other types, return as is

    # Fixed cache state cloning function
    def clone_cache_state(cache_state):
        cloned = []
        for (k, v) in cache_state:
            k_clone = k.detach().clone() if k is not None else None
            v_clone = v.detach().clone() if v is not None else None
            cloned.append((k_clone, v_clone))
        return cloned

    # Start the two-phase inference process
    start_time = time.time()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            # Phase 1: Generate initial patch with context
            f_hat, next_token_map, rng_state, cur_L, cond_BD, cache_states = (
                infer_model.generate_initial_patch_with_context(
                    B=tensor1.size(0),
                    label_B=tensor1,
                    label_B2=tensor2,
                    g_seed=args.seed,
                    cfg=args.cfg,
                    more_smooth=args.more_smooth,
                    context_position_ids=position_ids1,
                    context_mask=mask1,
                )
            )

            cache_state_clone = clone_cache_state(cache_states)
            rng_clone = copy_rng_state(rng_state)
            f_hat_clone = f_hat.clone()
            next_token_map_clone = next_token_map.clone()
            
            # Print status
            print(f"Initial patch generated. Continuing with remaining patches...")
            
            # Phase 2: Continue with remaining patches and MaskGIT final stage
            output_img = infer_model.continue_autoregressive_infer_cfg(
                B=tensor2.size(0),
                g_seed=None,  # Don't re-seed, use the state from phase 1
                label_B=tensor2,
                cfg=args.cfg,
                more_smooth=args.more_smooth,
                final_stage=0,
                num_maskgit_iters=1,
                init_f_hat=f_hat_clone,
                init_next_token_map=next_token_map_clone,
                init_rng_state=rng_clone,
                init_cur_L=cur_L,
                init_cond_BD=cond_BD,
                init_cache_state=cache_state_clone,
                context_position_ids=position_ids2,
                context_mask=mask2,
            )
    
    total_time = time.time() - start_time
    print(f"Generated image in {total_time:.2f}s.")

    prompts = [prompt]

    save_images(output_img.clone(), args.sample_folder_dir, args.store_seperately, prompts)
    print(f"Image saved to {args.sample_folder_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/scratch/dps9998/hart_inference/hart-0.7b-1024px/llm/")
    parser.add_argument("--text_model_path", type=str, default="/scratch/dps9998/hart_inference/Qwen2-VL-1.5B-Instruct/")
    parser.add_argument("--summary", type=str, default="boat")
    parser.add_argument("--prompt", type=str, default="a boat sailing in the pitch black night", help="Single prompt for image generation.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--more_smooth", type=bool, default=True)
    parser.add_argument("--sample_folder_dir", type=str, default="samples/")
    parser.add_argument("--store_seperately", action="store_true")
    args = parser.parse_args()
    main(args)
