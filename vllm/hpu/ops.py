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

import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as htorch
from typing import List, Optional, Tuple

def silu_and_mul(output, input):
    htorch.core.mark_step()
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)
    htorch.core.mark_step()

def gelu_new(output, input):
    raise NotImplementedError

def gelu_fast(output, input):
    raise NotImplementedError

def paged_attention_v1(query_in, key_cache_in, value_cache_in, head_mapping, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, attn_masks=None)  -> None:
    query = query_in.bfloat16()
    key_cache = key_cache_in.bfloat16()
    value_cache = value_cache_in.bfloat16()
    num_kv_heads = value_cache[0].shape[0]
    head_size = value_cache[0].shape[1]
    block_size = value_cache[0].shape[2]
    num_seqs = query.shape[0]
    num_query_heads = query.shape[1]
    max_num_blocks_per_seq = block_tables.shape[1]

    if alibi_slopes or num_query_heads != num_kv_heads: #or attn_masks is None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError

    attn_weights_blocks = []
    value_blocks = []
    seq_index = torch.tensor([0], dtype=torch.int64, device="hpu")

    for i in range(0, max_num_blocks_per_seq):
        # hard override for filler. These blocks would contribute nothing to the output due to zero attention_probs and will clog up compute resources
        if (i - 2) * block_size > torch.max(context_lens):
            break
        attn_weights = torch.full((num_seqs, num_query_heads, 1, block_size), torch.finfo(query.dtype).min, dtype=query.dtype, device="hpu")
        values = torch.zeros((num_seqs, num_query_heads, head_size, block_size), dtype=query.dtype, device="hpu")
        for seq_id in range(num_seqs):
            seq_index.fill_(seq_id)
            if i * block_size < context_lens[seq_id]:

                q =  torch.index_select(query, 0, seq_index).transpose(0, 1)
                key = torch.index_select(key_cache, 0, block_tables[seq_id][i]).squeeze(0)
                attn_weight = scale * torch.matmul(q, key)

                if attn_masks is not None:
                    attn_mask = torch.index_select(attn_masks[i], 0, seq_index)
                    attn_weight = torch.masked_fill(attn_weight, ~(attn_mask.unsqueeze(0).to(torch.bool)), torch.finfo(attn_weight.dtype).min)

                if context_lens[seq_id] < (i + 1) * block_size:
                    if context_lens[seq_id] - i*block_size < 0:
                        attn_weight = torch.finfo(query.dtype).min
                    else:
                        attn_weight[:, :, context_lens[seq_id] - i*block_size:] = torch.finfo(query.dtype).min
                attn_weights.index_copy_(0, seq_index, attn_weight.unsqueeze(0))
            #attn_weights[attn_weights == 0.0] = torch.finfo(query.dtype).min
            #if (i - 2) * block_size < max_context_len:
            value = torch.index_select(value_cache, 0, block_tables[seq_id][i])
            value = torch.nan_to_num(value)
            value[value < -1.0e+30] = 0.0
            values.index_copy_(0, seq_index, value)
            torch.hpu.synchronize()

        attn_weights_blocks.append(attn_weights.reshape(num_seqs * num_query_heads, 1, block_size))
        value_blocks.append(values.reshape(num_seqs * num_query_heads, head_size, block_size).transpose(1, 2))

    exp_sum = torch.zeros((*attn_weights_blocks[0].shape[:2], 1), dtype=attn_weights_blocks[0].dtype, device="hpu")
    for x in attn_weights_blocks:
        exp_sum.add_(torch.exp(x).sum(dim=-1, keepdim=True))

    output = torch.zeros_like(query)
    for i in range(len(attn_weights_blocks)):
        attention_probs = torch.exp(attn_weights_blocks[i]) / exp_sum
        value = value_blocks[i]
        out = torch.matmul(attention_probs.to(value.dtype), value).reshape(num_seqs, num_query_heads, head_size)
        output.add_(out)
    htorch.core.mark_step()
    return output.to(dtype=query_in.dtype)

def rms_norm(out, hidden_states, weight, eps):
    htorch.core.mark_step()
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    out.copy_(weight * hidden_states.to(input_dtype))
    htorch.core.mark_step()

def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style):
    raise NotImplementedError
    # update query and key in-place
    num_tokens = query.shape[0]
    num_heads = query.shape[-1] // head_size
    query = query.view(num_tokens, num_heads, head_size)
    key = key.view(num_tokens, num_heads, head_size)
    cos, sin = torch.split(cos_sin_cache, cos_sin_cache.shape[-1] // 2, dim=-1)
    if is_neox_style:
        sin = torch.cat((sin, sin), dim=-1)
        cos = torch.cat((cos, cos), dim=-1)
    else:
        sin = torch.repeat_interleave(sin, 2, -1)
        cos = torch.repeat_interleave(cos, 2, -1)

    query_rot = query[..., :head_size]
    query_pass = query[..., head_size:]
    key_rot = key[..., :head_size]
    key_pass = key[..., head_size:]

    query_rot = query_rot.transpose(0, 1)
    key_rot = key_rot.transpose(0, 1)
    cos = F.embedding(positions, cos)
    sin = F.embedding(positions, sin)

    query_rot, key_rot = apply_rope(query_rot, key_rot, cos, sin,
                                    is_neox_style)
    query_rot = query_rot.transpose(0, 1).contiguous()
    key_rot = key_rot.transpose(0, 1).contiguous()

    query.copy_(torch.cat((query_rot, query_pass), dim=-1))
    key.copy_(torch.cat((key_rot, key_pass), dim=-1))
    htorch.core.mark_step()

    # Output query/key shape: [num_tokens, num_tokens, head_size]
    return query, key
    #raise NotImplementedError

def awq_gemm(*args):
    raise NotImplementedError
