import torch

# vllm imports
from vllm import cache_ops
from vllm import attention_ops

_PARTITION_SIZE = 512


def reshape_and_cache(key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                      slots: torch.Tensor):
    cache_ops.reshape_and_cache(
        key, value, key_cache, value_cache, slots
    )


def attention(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_head_mapping: torch.Tensor,
        softmax_scale: float,
        block_tables: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
):
    # Adapted from: https://github.com/vllm-project/vllm/blob/f8a1e39fae05ca610be8d5a78be9d40f5274e5fc/vllm/model_executor/layers/attention.py
    # Copyright 2023 The vLLM team. All rights
    # reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #

    # value_cache => [num_blocks, num_heads, head_size, block_size]
    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (
            (max_s + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    use_v1 = max_num_partitions == 1 or num_seqs * num_heads > 512
    if use_v1:
        attention_ops.paged_attention_v1(
            out,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=out.device,
        )
        max_logits = torch.empty_like(exp_sums)
        attention_ops.paged_attention_v2(
            out,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
        )
