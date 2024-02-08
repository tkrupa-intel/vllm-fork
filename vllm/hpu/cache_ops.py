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

from typing import Tuple
import torch
import habana_frameworks.torch as htorch


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, is_prompt=False):
    """
    key: [num_tokens, num_heads, head_size]
    value: [num_tokens, num_heads, head_size]
    key_cache: [num_heads, head_size, block_size] * num_blocks
    value_cache: [num_heads, head_size, block_size] * num_blocks
    slot_mapping: [num_tokens]
    """
    num_tokens = key.shape[0]
    block_size = key_cache.shape[-1]
    slot_mapping = slot_mapping.to(key.device)
    # block_idx_list = [int(slot_idx / block_size) if slot_idx > 0 else slot_idx for slot_idx in slot_mapping.tolist()]
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    if is_prompt:
        # indices = torch.tensor([i for i in range(0, block_size)], device=key.device)
        for i in range(0, num_tokens, block_size):
            # if block_idx_list[i] < 0:
            #     # indices.add_(block_size)
            #     continue
            key_cache.index_put_([block_indices[i]], key[i:i+block_size].transpose(0,1).transpose(1,2))
            value_cache.index_put_([block_indices[i]], value[i:i+block_size].transpose(0,1).transpose(1,2))
            # key_cache.index_put_([block_indices[i]], key.index_select(0, indices).transpose(0,1).transpose(1,2))
            # value_cache.index_put_([block_indices[i]], value.index_select(0, indices).transpose(0,1).transpose(1,2))
            # indices.add_(block_size)
    else:
        # print(key_cache.data_ptr(), key_cache.shape)
        # print(key_cache[2, :, :, 2])
        key_cache = key_cache.permute(0, 3, 1, 2)
        value_cache = value_cache.permute(0, 3, 1, 2)
        # print(key_cache.data_ptr(), key_cache.shape)
        # print(key_cache[2, 2, :, :])
        block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_offsets = torch.fmod(slot_mapping, block_size)
        slot_indices = torch.stack([block_indices, block_offsets], dim=-1)
        index = torch.tensor(0, device=key.device)
        for i in range(num_tokens):
            key_cache[slot_indices[i][0], slot_indices[i][1], :, :] = key[i] # key.index_select(0, index)
            value_cache[slot_indices[i][0], slot_indices[i][1], :, :] = value[i] # value.index_select(0, index)
            # key_cache.index_put_([slot_indices[i]],  key[i])
            # value_cache.index_put_([slot_indices[i]], value[i])
            # key_cache.index_put_([slot_indices[i]],  key.index_select(0, index))
            # value_cache.index_put_([slot_indices[i]], value.index_select(0, index))
            index.add_(1)
        # print(key_cache.data_ptr(), key_cache.shape)
        key_cache = key_cache.permute(0, 2, 3, 1)
        value_cache = value_cache.permute(0, 2, 3, 1)
        # print(key_cache.data_ptr(), key_cache.shape)


def swap_blocks(src, dst, block_mapping):
    #print(f'swap_blocks: {block_mapping}')
    index_src = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    index_dst = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    for src_idx, dst_idx in block_mapping.items():
        index_src[0] = src_idx
        index_dst[0] = dst_idx
        dst.index_put_([index_dst], src.index_select(0, index_src))
        if dst.device.type == 'hpu':
            htorch.core.mark_step()
            torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, block_mapping):
    #print(f'copy_blocks: {block_mapping}')
    index_src = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    index_dst = torch.zeros((1,), dtype=torch.int32, device=key_caches[0].device)
    for src, dsts in block_mapping.items():
        index_src[0] = src
        for dst in dsts:
            index_dst[0] = dst
            for key_cache in key_caches:
                key_cache.index_copy_(0, index_dst, key_cache.index_select(0, index_src))
            for value_cache in value_caches:
                value_cache.index_copy_(0, index_dst, value_cache.index_select(0, index_src))
        if key_caches[0].device.type == 'hpu':
            htorch.core.mark_step()


'''
def create_cache_view(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, num_heads, head_size, block_size = key_cache.shape
    cache_stride = key_cache.stride()
    cache_offset = key_cache.storage_offset()
    block_shape = (1, num_heads, head_size, block_size)
    block_offset = block_idx * (cache_stride[-1] * cache_stride[-2] * cache_stride[-3])
    key_block = torch.as_strided(key_cache,
                                block_shape,
                                cache_stride,
                                cache_offset+block_offset).squeeze(0)
    value_block = torch.as_strided(value_cache,
                                    block_shape,
                                    cache_stride,
                                    cache_offset+block_offset).squeeze(0)
    return key_block, value_block


def reshape_and_cache_backup1(key, value, key_cache, value_cache, slot_mapping, is_prompt=False):
    """
    key: [num_tokens, num_heads, head_size]
    value: [num_tokens, num_heads, head_size]
    key_cache: [num_heads, head_size, block_size] * num_blocks
    value_cache: [num_heads, head_size, block_size] * num_blocks
    slot_mapping: [num_tokens]
    """
    block_size = key_cache[0].shape[2]
    block_idx_list = [int(slot_idx / block_size) if slot_idx > 0 else slot_idx for slot_idx in slot_mapping.tolist()]
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    if is_prompt:
        indices = torch.tensor([i for i in range(0, block_size)], device=key.device)
        for i in range(0, len(block_idx_list), block_size): # for i in range(0, block_indices.shape[0], block_size):
            if block_idx_list[i] < 0:
                continue
            block_idx_tensor = block_indices.index_select(0, torch.tensor(i, device=key.device))
            key_cache.index_put_([block_idx_tensor], key.index_select(0, indices).transpose(0,1).transpose(1,2))
            value_cache.index_put_([block_idx_tensor], value.index_select(0, indices).transpose(0,1).transpose(1,2))
            indices.add_(block_size)
    else:
        block_idx_list = [int(slot_idx / block_size) if slot_idx > 0 else slot_idx for slot_idx in slot_mapping.tolist()]
        block_offset_list = [slot_idx % block_size for slot_idx in slot_mapping.tolist()]
        index = torch.tensor(0, device=key.device)
        for block_idx, block_offset in zip(block_idx_list, block_offset_list):
            key_block, value_block = create_cache_view(key_cache, value_cache, block_idx)
            slot_idx = torch.tensor(block_offset, device=key.device)
            key_block.index_copy_(-1, slot_idx, key.index_select(0, index).transpose(0,1).transpose(1,2))
            value_block.index_copy_(-1, slot_idx, value.index_select(0, index).transpose(0,1).transpose(1,2))
            index.add_(1)


def reshape_and_cache_backup2(key, value, key_cache, value_cache, slot_mapping, is_prompt=False):
    """
    key: [num_tokens, num_heads, head_size]
    value: [num_tokens, num_heads, head_size]
    key_cache: [num_heads, head_size, block_size] * num_blocks
    value_cache: [num_heads, head_size, block_size] * num_blocks
    slot_mapping: [num_tokens]
    """
    block_size = key_cache[0].shape[2]
    block_idx_list = [int(slot_idx / block_size) if slot_idx > 0 else slot_idx for slot_idx in slot_mapping.tolist()]
    if is_prompt:
        cached_set = set()
        indices = torch.tensor([i for i in range(0, block_size)], device=key.device)
        for block_idx in block_idx_list:
            if block_idx in cached_set or block_idx < 0:
                continue
            else:
                cached_set.add(block_idx)
                key_block, value_block = create_cache_view(key_cache, value_cache, block_idx)
                key_block.copy_(key.index_select(0, indices).transpose(0,1).transpose(1,2))
                value_block.copy_(value.index_select(0, indices).transpose(0,1).transpose(1,2))
            indices.add_(block_size)
    else:
        block_offset_list = [slot_idx % block_size for slot_idx in slot_mapping.tolist()]
        index = torch.tensor(0, device=key.device)
        # slot_idx = torch.tensor(0, device=key.device)
        for block_idx, block_offset in zip(block_idx_list, block_offset_list):
            key_block, value_block = create_cache_view(key_cache, value_cache, block_idx)
            # slot_idx.copy_(block_offset)
            slot_idx = torch.tensor(block_offset, device=key.device)
            key_block.index_copy_(-1, slot_idx, key.index_select(0, index).transpose(0,1).transpose(1,2))
            value_block.index_copy_(-1, slot_idx, value.index_select(0, index).transpose(0,1).transpose(1,2))
            index.add_(1)
'''