"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest
from vllm.utils import is_hip, is_hpu


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = ModelRunner(model_config,
                                        parallel_config,
                                        scheduler_config,
                                        device_config,
                                        lora_config=self.lora_config,
                                        kv_cache_dtype=kv_cache_dtype,
                                        is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    def init_model(self, cupy_port: Optional[int] = None) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        elif is_hpu():
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device("hpu")
            torch.cuda.set_device(self.device)
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     cupy_port, self.distributed_init_method)
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            if sampling_params.prompt_logprobs is not None:
                # NOTE: prompt token positions do not need sample, skip
                categorized_sample_indices_start_idx += prompt_len - 1

            categorized_sample_indices[sampling_params.sampling_type].append(
                categorized_sample_indices_start_idx)
            categorized_sample_indices_start_idx += 1

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([0] * prompt_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        max_seq_len = max(prompt_lens) if prompt_lens else 1
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                # We need to do this in this loop as we need to know max_seq_len
                assert len(
                    seq_ids) == 1, "Prompt input should have only one seq."
                sampling_params = seq_group_metadata.sampling_params
                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += max_seq_len
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            num_seqs = len(seq_ids)
            selected_token_indices.extend(
                range(selected_token_start_idx,
                      selected_token_start_idx + num_seqs))
            selected_token_start_idx += num_seqs

            categorized_sample_indices[sampling_params.sampling_type].extend(
                range(categorized_sample_indices_start_idx,
                      categorized_sample_indices_start_idx + num_seqs))
            categorized_sample_indices_start_idx += num_seqs

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                context_len = seq_data.get_len()
                position = context_len - 1
                if self.sliding_window is not None:
                    context_len = min(context_len, self.sliding_window)
                input_positions.append([position])

                block_table = seq_group_metadata.block_tables[seq_id]

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                generation_block_tables.append(block_table)

        def round_up(n, multiple):
            return (n + multiple - 1) // multiple * multiple

        if self.block_size is not None:
            if max_seq_len != 1:
                max_seq_len = round_up(max_seq_len, self.block_size)

        padded_input_tokens = [
            _pad_to_max(tokens, max_seq_len, pad=0) for tokens in input_tokens
        ]
        padded_input_positions = [
            _pad_to_max(positions, max_seq_len, pad=0)
            for positions in input_positions
        ]
        padded_slot_mapping = [
            _pad_to_max(mapping, max_seq_len, pad=-1)
            for mapping in slot_mapping
        ]
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq, pad=0)
            for block_table in generation_block_tables
        ]

        # Convert to tensors.
        tokens_tensor = torch.tensor(padded_input_tokens,
                                     dtype=torch.long,
                                     device="cuda")
        positions_tensor = torch.tensor(padded_input_positions,
                                        dtype=torch.long,
                                        device="cuda")
        slot_mapping_tensor = torch.tensor(padded_slot_mapping,
                                           dtype=torch.long,
                                           device="cpu")
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device="cpu")
        selected_token_indices = torch.tensor(selected_token_indices,
                                              dtype=torch.long,
                                              device="cuda")
        categorized_sample_indices = {
            t: torch.tensor(seq_ids, dtype=torch.int, device="cuda")
            for t, seq_ids in categorized_sample_indices.items()
        }
        block_tables_tensor = torch.tensor(padded_block_tables,
                                           dtype=torch.int,
                                           device="cpu")

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            sliding_window=self.sliding_window,
        )

        # Create attention mask
        if max_num_blocks_per_seq != 0:
            attn_masks = torch.zeros((max_num_blocks_per_seq, len(input_tokens), self.block_size), dtype=torch.int64)
            for i in range(0, max_num_blocks_per_seq):
                for seq_id in range(len(input_tokens)):
                    if (i * self.block_size) < context_lens[seq_id] and (i + 1) * self.block_size > context_lens[seq_id]:
                        attn_masks[i][seq_id, :context_lens[seq_id] % self.block_size] = 1
                    elif (i + 1) * self.block_size <= context_lens[seq_id]:
                        attn_masks[i][seq_id, :] = 1
            input_metadata.attention_masks = attn_masks.to(device="cuda")
        return tokens_tensor, positions_tensor, input_metadata

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()


def init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    cupy_port: Optional[int],
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    if cupy_utils.is_initialized():
        cupy_world_size = cupy_utils.get_world_size()
        if cupy_world_size != parallel_config.world_size:
            raise RuntimeError(
                "cupy.distributed is already initialized but the cupy world "
                "size does not match parallel_config.world_size "
                f"({cupy_world_size} vs. {parallel_config.world_size}).")
    elif (parallel_config.world_size > 1 and cupy_port is not None
          and not is_hip()):
        # NOTE(woosuk): We don't initialize CuPy process group when world size
        # is 1.
        # TODO(woosuk): Support multi-node connection.
        cupy_utils.init_process_group(
            world_size=parallel_config.world_size,
            rank=rank,
            host="localhost",
            port=cupy_port,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    if cupy_utils.is_initialized():
        cupy_utils.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    # Initialize a custom fast all-reduce implementation.
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
