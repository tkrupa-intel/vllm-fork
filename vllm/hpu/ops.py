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
    device = 'cpu'
    query = query_in.bfloat16().to(device)
    key_cache = key_cache_in.bfloat16().to(device)
    value_cache = value_cache_in.bfloat16().to(device)
    block_tables = block_tables.to(device)
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes.to(device)
    if attn_masks is not None:
        attn_masks = attn_masks.to(device)
    context_lens = context_lens.to(device)
    num_kv_heads = value_cache[0].shape[0]
    head_size = value_cache[0].shape[1]
    block_size = value_cache[0].shape[2]
    num_seqs = query.shape[0]
    num_query_heads = query.shape[1]
    max_num_blocks_per_seq = block_tables.shape[1]

    num_queries_per_kv = num_query_heads // num_kv_heads
    attn_weights_blocks = torch.full((max_num_blocks_per_seq, num_seqs * num_query_heads, 1, block_size), torch.finfo(query.dtype).min, dtype=query.dtype, device=device)
    value_blocks = torch.zeros((max_num_blocks_per_seq, num_seqs * num_kv_heads, block_size, head_size), dtype=query.dtype, device=device)
    seq_index = torch.tensor([0], dtype=torch.int64, device=device)

    for i in range(0, max_num_blocks_per_seq):
        # hard override for filler. These blocks would contribute nothing to the output due to zero attention_probs and will clog up compute resources
        if (i - 2) * block_size > torch.max(context_lens):
            break
        attn_weights = torch.full((num_seqs, num_query_heads, 1, block_size), torch.finfo(query.dtype).min, dtype=query.dtype, device=device)
        values = torch.zeros((num_seqs, num_kv_heads, head_size, block_size), dtype=query.dtype, device=device)
        for seq_id in range(num_seqs):
            seq_index.fill_(seq_id)
            if i * block_size < context_lens[seq_id]:

                q =  torch.index_select(query, 0, seq_index).transpose(0, 1)
                key = torch.index_select(key_cache, 0, block_tables[seq_id][i]).squeeze(0)
                if num_queries_per_kv > 1:
                    # Handle MQA and GQA
                    key = torch.repeat_interleave(key, num_queries_per_kv, dim=0)
                attn_weight = scale * torch.matmul(q, key)

                if attn_masks is not None:
                    attn_mask = torch.index_select(attn_masks[i], 0, seq_index)
                    attn_weight = torch.masked_fill(attn_weight, ~(attn_mask.unsqueeze(0).to(torch.bool)), torch.finfo(attn_weight.dtype).min)

                if context_lens[seq_id] < (i + 1) * block_size:
                    if context_lens[seq_id] - i*block_size < 0:
                        attn_weight = torch.finfo(query.dtype).min
                    else:
                        attn_weight[:, :, context_lens[seq_id] - i*block_size:] = torch.finfo(query.dtype).min
                if alibi_slopes is not None:
                    # Create the ALiBi bias used in the paged attention kernel.
                    position_ids = torch.arange(context_lens[seq_id], device=device).int()
                    alibi_bias = (position_ids - context_lens[seq_id] + 1).float()
                    alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)
                    lower_bound = i*block_size
                    upper_bound = (i+1)*block_size
                    alibi_bias_slice = alibi_bias[:,:,lower_bound:upper_bound]
                    block_alibi_bias = torch.zeros((alibi_bias.shape[0], alibi_bias.shape[1], block_size), device=device, dtype=attn_weight.dtype)
                    block_alibi_bias[:, :, :alibi_bias_slice.shape[-1]] = alibi_bias_slice.to(attn_weight.dtype)
                    #if torch.exp(torch.add(attn_weight, block_alibi_bias)).sum(dim=-1, keepdim=True).isinf().any():                     
                    #    import pdb; pdb.set_trace()
                    attn_weight.add_(block_alibi_bias)
                    
                attn_weights.index_copy_(0, seq_index, attn_weight.unsqueeze(0))
            #attn_weights[attn_weights == 0.0] = torch.finfo(query.dtype).min
            #if (i - 2) * block_size < max_context_len:
            value = torch.index_select(value_cache, 0, block_tables[seq_id][i])
            value = torch.nan_to_num(value)
            value[value < -1.0e+30] = 0.0
            values.index_copy_(0, seq_index, value)
            if device == 'hpu':
                torch.hpu.synchronize()

        attn_weights_blocks[i] = attn_weights.reshape(num_seqs * num_query_heads, 1, block_size)
        value_blocks[i] = values.reshape(num_seqs * num_kv_heads, head_size, block_size).transpose(1, 2)

    from functools import reduce
    max_weight = torch.amax(attn_weights_blocks, dim=(0,-1)).unsqueeze(-1) #reduce(lambda x,y: torch.max(x,y), [torch.max(block) for block in attn_weights_blocks])
  #  import pdb; pdb.set_trace()
    #softmax_val = torch.nn.functional.softmax(attn_weights_blocks, dim=-1)
    exp_sum = torch.zeros((*attn_weights_blocks[0].shape[:2], 1), dtype=attn_weights_blocks[0].dtype, device=device)
    #import pdb; pdb.set_trace()
    for x in attn_weights_blocks:
        exp_sum.add_(torch.exp(x - max_weight).sum(dim=-1, keepdim=True))
    output = torch.zeros_like(query)
    for i in range(max_num_blocks_per_seq):
        attention_probs = torch.exp(attn_weights_blocks[i] - max_weight) / (exp_sum + 1e-9)
        #attention_probs = softmax_val[i]
        #import pdb; pdb.set_trace()
        value = value_blocks[i]
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            value_4d_view = value.reshape(num_seqs, num_kv_heads, block_size, head_size)
            value = torch.repeat_interleave(value_4d_view, num_queries_per_kv, dim=1).reshape(num_seqs * num_query_heads, block_size, head_size)
        out = torch.matmul(attention_probs.to(value.dtype), value).reshape(num_seqs, num_query_heads, head_size)
        output.add_(out)
    if device == 'hpu':
        htorch.core.mark_step()
    return output.to(dtype=query_in.dtype)



def paged_attention_v1_old(query_in, key_cache_in, value_cache_in, head_mapping, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, attn_masks=None)  -> None:
    device = 'cpu'
    query = query_in.bfloat16().to(device)
    key_cache = key_cache_in.bfloat16().to(device)
    value_cache = value_cache_in.bfloat16().to(device)
    block_tables = block_tables.to(device)
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes.to(device)
    if attn_masks is not None:
        attn_masks = attn_masks.to(device)
    context_lens = context_lens.to(device)
    num_kv_heads = value_cache[0].shape[0]
    head_size = value_cache[0].shape[1]
    block_size = value_cache[0].shape[2]
    num_seqs = query.shape[0]
    num_query_heads = query.shape[1]
    max_num_blocks_per_seq = block_tables.shape[1]

    num_queries_per_kv = num_query_heads // num_kv_heads
    attn_weights_blocks = []
    value_blocks = []
    seq_index = torch.tensor([0], dtype=torch.int64, device=device)

    for i in range(0, max_num_blocks_per_seq):
        # hard override for filler. These blocks would contribute nothing to the output due to zero attention_probs and will clog up compute resources
        if (i - 2) * block_size > torch.max(context_lens):
            break
        attn_weights = torch.full((num_seqs, num_query_heads, 1, block_size), torch.finfo(query.dtype).min, dtype=query.dtype, device=device)
        values = torch.zeros((num_seqs, num_kv_heads, head_size, block_size), dtype=query.dtype, device=device)
        for seq_id in range(num_seqs):
            seq_index.fill_(seq_id)
            if i * block_size < context_lens[seq_id]:

                q =  torch.index_select(query, 0, seq_index).transpose(0, 1)
                key = torch.index_select(key_cache, 0, block_tables[seq_id][i]).squeeze(0)
                if num_queries_per_kv > 1:
                    # Handle MQA and GQA
                    key = torch.repeat_interleave(key, num_queries_per_kv, dim=0)
                attn_weight = scale * torch.matmul(q, key)
                if alibi_slopes is not None:
                    # Create the ALiBi bias used in the paged attention kernel.
                    position_ids = torch.arange(context_lens[seq_id], device=device).int()
                    alibi_bias = (position_ids - context_lens[seq_id] + 1).float()
                    alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)
                    lower_bound = i*block_size
                    upper_bound = min([(i+1)*block_size, context_lens[seq_id]])
                    alibi_bias_slice = alibi_bias[:,:,lower_bound:upper_bound]
                    block_alibi_bias = torch.zeros((alibi_bias.shape[0], alibi_bias.shape[1], block_size), device=device, dtype=attn_weight.dtype)
                    block_alibi_bias[:, :, :alibi_bias_slice.shape[-1]] = alibi_bias_slice.to(attn_weight.dtype)
                    #if torch.exp(torch.add(attn_weight, block_alibi_bias)).sum(dim=-1, keepdim=True).isinf().any():                     
                    #    import pdb; pdb.set_trace()
                    attn_weight.add_(block_alibi_bias)
                    
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
            if device == 'hpu':
                torch.hpu.synchronize()

        attn_weights_blocks.append(attn_weights.reshape(num_seqs * num_query_heads, 1, block_size))
        value_blocks.append(values.reshape(num_seqs * num_kv_heads, head_size, block_size).transpose(1, 2))

    from functools import reduce
    max_weight = 0 #reduce(lambda x,y: torch.max(x,y), [torch.max(block) for block in attn_weights_blocks])
    exp_sum = torch.zeros((*attn_weights_blocks[0].shape[:2], 1), dtype=attn_weights_blocks[0].dtype, device=device)
    for x in attn_weights_blocks:
        exp_sum.add_(torch.exp(x - max_weight).sum(dim=-1, keepdim=True))
    output = torch.zeros_like(query)
    for i in range(len(attn_weights_blocks)):
        attention_probs = torch.exp(attn_weights_blocks[i] - max_weight) / (exp_sum + 1e-9)
        value = value_blocks[i]
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            value_4d_view = value.reshape(num_seqs, num_kv_heads, block_size, head_size)
            value = torch.repeat_interleave(value_4d_view, num_queries_per_kv, dim=1).reshape(num_seqs * num_query_heads, block_size, head_size)
        out = torch.matmul(attention_probs.to(value.dtype), value).reshape(num_seqs, num_query_heads, head_size)
        output.add_(out)
    if device == 'hpu':
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
