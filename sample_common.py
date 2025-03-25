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


def autoregressive_infer_cfg_1(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = 0,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        final_stage=0,
        num_maskgit_iters=1,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        # num_maskgit_iters = 1
        # final_stage = 2
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        assert label_B is not None
        assert label_B.shape[1] == self.context_token

        sos = cond_BD = self.context_embed(
            self.context_norm(
                torch.cat((label_B, torch.full_like(label_B, fill_value=0.0)), dim=0)
            )
        )
        # Haotian: need to handle CFG here so we replicate context position ids
        context_position_ids = torch.cat(
            (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
            dim=0,
        )

        b = context_mask.shape[0]
        context_mask = torch.cat(
            (context_mask, torch.full_like(context_mask, fill_value=0))
        )
        context_mask[b:, 0] = 1

        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L)

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

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        f_hat_store = []

        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums[:4]):  
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            if si > 0:
                cur_L += pn * pn
            else:
                cur_L += self.context_token
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            
            for b in self.blocks:
                # Haotian: si used for position embed
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
            # Haotian: Added for text-conditioned generation
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map  = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
            )

            f_hat_store.append(f_hat.clone())

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(
                2, 1, 1
            )  # double the batch sizes due to CFG


        for b in self.blocks:
            b.attn.kv_caching(False)                   
        return f_hat_store, h_BChw 

def autoregressive_infer_cfg_2(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = 0,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        patch_1_f_hat=None,
        patch_1_h_BChw=None,
        final_stage=0,
        num_maskgit_iters=1,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        # num_maskgit_iters = 1
        # final_stage = 2
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        assert label_B is not None
        assert label_B.shape[1] == self.context_token

        sos = cond_BD = self.context_embed(
            self.context_norm(
                torch.cat((label_B, torch.full_like(label_B, fill_value=0.0)), dim=0)
            )
        )
        # Haotian: need to handle CFG here so we replicate context position ids
        context_position_ids = torch.cat(
            (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
            dim=0,
        )

        b = context_mask.shape[0]
        context_mask = torch.cat(
            (context_mask, torch.full_like(context_mask, fill_value=0))
        )
        context_mask[b:, 0] = 1

        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L)

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

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks:
            b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums[:-1]):  # si: i-th segment
            
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            if si > 0:
                cur_L += pn * pn
            else:
                cur_L += self.context_token
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
    
            for b in self.blocks:
                # Haotian: si used for position embed
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
            # Haotian: Added for text-conditioned generation
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
            )

            if si < 4:
                f_hat = patch_1_f_hat[si]

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(
                2, 1, 1
            )  # double the batch sizes due to CFG

        ################ last stage maskgit ################
        si = len(self.patch_nums) - 1
        mask = torch.ones(B, self.last_level_pns).cuda()
        tokens = torch.zeros(B, self.last_level_pns, self.Cvae).cuda()
        orders = self.sample_orders(B)

        num_iter = num_maskgit_iters
        indices = list(range(num_iter))
        # generate latents with maskgit
        for step in indices:
            # mask_ratio = 1 - (step + 1) / num_iter
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.last_level_pns * mask_ratio)]).cuda()
            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            # get masking for next iteration and locations to be predicted in this iteration
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
                # Haotian: si used for position embed
                # note: m_maskgit makes sure that PEs are correct.
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
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
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

        for b in self.blocks:
            b.attn.kv_caching(False)
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

    # Get prompts: summary prompt for initial patch, regular prompt for details
    summary_prompt = args.summary_prompt
    
    detail_prompt = args.prompt
    

    print(f"Summary prompt (initial patch): '{summary_prompt}'")
    print(f"Detail prompt (remaining patches): '{detail_prompt}'")

    # Encode the summary prompt for initial patch
    summary_tokens, summary_mask, summary_position_ids, summary_tensor = encode_prompts(
        [summary_prompt],
        text_model,
        text_tokenizer,
        args.max_token_length,
        llm_system_prompt,
        args.use_llm_system_prompt,
    )

    # Encode the detail prompt for remaining patches
    detail_tokens, detail_mask, detail_position_ids, detail_tensor = encode_prompts(
        [detail_prompt],
        text_model,
        text_tokenizer,
        args.max_token_length,
        llm_system_prompt,
        args.use_llm_system_prompt,
    )

    # Prepare for inference
    infer_model = ema_model if args.use_ema else model
    if not hasattr(infer_model, "autoregressive_infer_cfg_1"):
        from types import MethodType
        infer_model.autoregressive_infer_cfg_1 = MethodType(autoregressive_infer_cfg_1, infer_model)

    if not hasattr(infer_model, "autoregressive_infer_cfg_2"):
        from types import MethodType
        infer_model.autoregressive_infer_cfg_2 = MethodType(autoregressive_infer_cfg_2, infer_model)


    
    # Create output directory if it doesn't exist
    os.makedirs(args.sample_folder_dir, exist_ok=True)

    
    # Start the two-phase inference process with different prompts
    start_time = time.time()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            # Phase 1: Generate initial patch with context from summary prompt
            f_hat, h_BChw = (
                infer_model.autoregressive_infer_cfg_1(
                    B=summary_tensor.size(0),
                    label_B=summary_tensor,
                    g_seed=0,
                    cfg=args.cfg,
                    more_smooth=args.more_smooth,
                    context_position_ids=summary_position_ids,
                    context_mask=summary_mask,
                    )
            )

            output_img = (
                infer_model.autoregressive_infer_cfg_2(
                    B=detail_tensor.size(0),
                    label_B=detail_tensor,
                    cfg=args.cfg,
                    g_seed=0,
                    more_smooth=args.more_smooth,
                    context_position_ids=detail_position_ids,
                    context_mask=detail_mask,
                    patch_1_f_hat=f_hat,
                    patch_1_h_BChw=h_BChw,
                    )
            )
            
    total_time = time.time() - start_time
    print(f"Generated image in {total_time:.2f}s.")

    # Save the output image with information about both prompts
    # Create a list with the two prompts for metadata
    prompts = [f"Summary: {summary_prompt} | Detail: {detail_prompt}"]
    
    # Use the existing save_images function
    save_images(output_img.clone(), args.sample_folder_dir, args.store_seperately, prompts)
    print(f"Image saved to {args.sample_folder_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/scratch/dps9998/hart_inference/hart-0.7b-1024px/llm/")
    parser.add_argument("--text_model_path", type=str, default="/scratch/dps9998/hart_inference/Qwen2-VL-1.5B-Instruct/")
    parser.add_argument("--summary_prompt", type=str, default="dog", 
                       help="Summary prompt for initial patch generation.")
    parser.add_argument("--prompt", type=str, default="a golden retriever waggling its tail", 
                       help="Detail prompt for remaining patches.")
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
