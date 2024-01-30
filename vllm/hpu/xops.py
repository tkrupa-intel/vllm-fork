###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################


import habana_frameworks.torch as htorch
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from .attn_bias import AttentionBias


# # xops.memory_efficient_attention_forward
# def memory_efficient_attention_forward(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attn_bias = None,
#     p: float = 0.0,
#     scale: Optional[float] = None
# ) -> torch.Tensor:
#     # scale = 1 / query.shape[-1] ** 0.5
#     query = query * scale
#     attn = query @ key.transpose(-2, -1)
#     if attn_bias is not None:
#         shape=(query.shape[0], query.shape[1], query.shape[-2], query.shape[-2])
#         attn_mask = torch.full(shape, dtype=query.dtype, fill_value=float("-inf"), device=query.device)
#         attn_mask = torch.triu(attn_mask, diagonal=1).to(query.dtype)
#         attn = attn + attn_mask
#     attn = attn.softmax(-1)
#     attn = torch.nn.functional.dropout(attn, p)
#     return attn @ value


def block_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = query.transpose(0,1) @ key.transpose(0, 1).transpose(1, 2)
    #print(f'q: {query.shape}, q^T: {query.transpose(0,1).shape}, k: {key.shape}, k^T: {key.transpose(0, 1).transpose(1, 2).shape}, v: {value.shape}, v^T: {value.transpose(0, 1).shape}, attn: {attn.shape}, attn_mask: {attn_mask.shape}')
    if attn_mask is not None:
        attn = attn + attn_mask.to(attn.dtype)
    attn = attn.softmax(-1)
    out = attn @ value.transpose(0, 1)
    out = out.transpose(0, 1)
    return out


def memory_efficient_attention_forward(
    query: torch.Tensor, # [B, Mq, H, K] or [B, Mq, K] when H=1 or [B, Mq, G, H, K] for MQA/GQA
    key: torch.Tensor,   # [B, Mkv, H, K] or [B, Mkv, K] when H=1 or [B, Mkv, G, H, K] for MQA/GQA
    value: torch.Tensor, # [B, Mkv, H, K] or [B, Mkv, K] when H=1 or [B, Mkv, G, H, K] for MQA/GQA
    cu_seq_lens: List[int],
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    dim = query.dim()
    # dim sanity checks, remove me later
    if dim == 3: # MHA with H=1
        batch_q, seq_len_q, embed_size_q = query.shape
        batch_k, seq_len_k, embed_size_k = key.shape
        batch_v, seq_len_v, embed_size_v = value.shape
        assert batch_q == batch_k and batch_q == batch_v and batch_k == batch_v
        assert seq_len_q == seq_len_v and seq_len_k == seq_len_v
        assert embed_size_q == embed_size_k and embed_size_q == embed_size_v and embed_size_k == embed_size_v
        print(f'memory_efficient_attention_forward: 3d MHA dim check successful: B={batch_q}, Mq={seq_len_q}, Mkv={seq_len_k}, K={embed_size_q}')

    if dim == 4: # MHA with variable num_heads
        batch_q, seq_len_q, num_heads_q, embed_size_q = query.shape
        batch_k, seq_len_k, num_heads_k, embed_size_k = key.shape
        batch_v, seq_len_v, num_heads_v, embed_size_v = value.shape
        assert batch_q == batch_k and batch_q == batch_v and batch_k == batch_v
        assert seq_len_k == seq_len_v
        assert num_heads_q == num_heads_k and num_heads_q == num_heads_k and num_heads_k == num_heads_v
        assert embed_size_q == embed_size_k and embed_size_q == embed_size_v and embed_size_k == embed_size_v
        print(f'memory_efficient_attention_forward: 4d MHA dim check successful: B={batch_q}, Mq={seq_len_q}, Mkv={seq_len_k}, H={num_heads_q}, K={embed_size_q}')

    if dim == 5: # MQA/GQA
        batch_q, seq_len_q, num_head_groups_q, num_heads_q, embed_size_q = query.shape
        batch_k, seq_len_k, num_head_groups_k, num_heads_k, embed_size_k = key.shape
        batch_v, seq_len_v, num_head_groups_v, num_heads_v, embed_size_v = value.shape
        assert batch_q == batch_k and batch_q == batch_v and batch_k == batch_v
        assert seq_len_k == seq_len_v
        assert num_heads_q == num_heads_k and num_heads_q == num_heads_k and num_heads_k == num_heads_v
        assert embed_size_q == embed_size_k and embed_size_q == embed_size_v and embed_size_k == embed_size_v
        assert num_head_groups_q == num_head_groups_k and num_head_groups_q == num_head_groups_v and num_head_groups_k == num_head_groups_v
        print(f'memory_efficient_attention_forward: 5d MQA/GQA dim check successful: B={batch_q}, Mq={seq_len_q}, Mkv={seq_len_k}, G={num_head_groups_q}, H={num_heads_q}, K={embed_size_q}')

    B = query.shape[0]
    Mq = query.shape[1]
    Mkv = key.shape[1]
    G = None if dim != 5 else query.shape[2]
    H = 1 if dim == 3 else query.shape[-2]
    K = query.shape[-1]

    if dim == 5:
        query, key, value = query.squeeze(2).squeeze(0), key.squeeze(2).squeeze(0), value.squeeze(2).squeeze(0)    
    if dim == 4:
        query, key, value = query.squeeze(0), key.squeeze(0), value.squeeze(0)
    
    num_seqs = len(cu_seq_lens) - 1
    outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx
        mask_start_idx = i * seq_len
        mask_end_idx = (i + 1) * seq_len

        # # Create attention mask.
        # attn_mask = torch.ones(seq_len, seq_len, dtype=query.dtype)
        # attn_mask[:seq_lens[i],:seq_lens[i]] = torch.triu(
        #     attn_mask[:seq_lens[i],:seq_lens[i]],
        #     diagonal=1
        # )
        # attn_mask = attn_mask * -10000.0 # torch.finfo(query.dtype).min
        # attn_mask = attn_mask.to(dtype=query.dtype, device=query.device)

        attn_mask = attn_bias.materialize(device=query.device)
        #print(attn_mask.shape, attn_mask[mask_start_idx:mask_end_idx, mask_start_idx:mask_end_idx].shape)
        output = block_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask[mask_start_idx:mask_end_idx,
                                mask_start_idx:mask_end_idx], # attn_mask=attn_mask,
        )
        outputs.append(output)
    out = torch.cat(outputs, dim=0)
    if dim == 4:
        out = out.reshape(B, Mq, H, K)
    elif dim == 5:
        out = out.reshape(B, Mq, G, H, K)
    return out
