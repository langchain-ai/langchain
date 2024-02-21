# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig
from typing import Optional, List, Tuple

from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.utils.flash_attn import attention
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    FastLayerNorm,
    PositionRotaryEmbedding,
    get_linear,
)


def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelRowLinear(linear, process_group=weights.process_group)


def load_qkv(config, prefix: str, weights, num_heads, head_size, hidden_size):
    weight = weights.get_multi_weights_col([prefix], quantize=config.quantize, dim=0)
    if isinstance(weight, torch.Tensor):
        # Only on non quantized versions
        weight = (
            weight.view(
                num_heads,
                3,
                head_size,
                hidden_size,
            )
            .permute(1, 0, 2, 3)
            .reshape(-1, hidden_size)
        )

    bias = weights.get_sharded(f"{prefix}.bias", dim=0)
    bias = bias.view(num_heads, 3, head_size).permute(1, 0, 2).reshape(-1)

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelColumnLinear(linear)


class FlashNeoxAttention(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()

        self.rotary_emb = PositionRotaryEmbedding.load(
            config=config, prefix=f"{prefix}.rotary_emb", weights=weights
        )

        self.softmax_scale = self.head_size ** (-0.5)

        self.query_key_value = load_qkv(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            num_heads=self.num_heads,
            head_size=self.head_size,
            hidden_size=self.hidden_size,
        )
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=True
        )
        self.kv_head_mapping = torch.arange(
            0, self.num_heads, dtype=torch.int32, device=weights.device
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)

        # Inplace rotary
        self.rotary_emb(qkv[:, 0], qkv[:, 1], cos, sin)

        paged_attention.reshape_and_cache(
            qkv[:, 1], qkv[:, 2], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(qkv[:, 0])

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            paged_attention.attention(
                attn_output,
                qkv[:, 0],
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class FlashMLP(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = load_row(
            config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashNeoXLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()

        layer_norm_eps = config.layer_norm_eps

        prefix = f"gpt_neox.layers.{layer_id}"

        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=layer_norm_eps
        )
        self.post_attention_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=layer_norm_eps,
        )
        self.attention = FlashNeoxAttention(
            config, prefix=f"{prefix}.attention", weights=weights
        )

        self.mlp = FlashMLP(config, prefix=f"{prefix}.mlp", weights=weights)
        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        if self.use_parallel_residual:
            ln1_hidden_states, _ = self.input_layernorm(hidden_states)

            attn_output = self.attention(
                ln1_hidden_states,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache,
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

            ln2_hidden_states, _ = self.post_attention_layernorm(hidden_states)

            mlp_output = self.mlp(ln2_hidden_states)
            intermediate = mlp_output + attn_output

            if self.process_group.size() > 1:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate + hidden_states, None
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.attention(
                hidden_states,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache,
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashGPTNeoXPreTrainedModel(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = False
    _no_split_modules = None


class FlashGPTNeoXModel(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.embed_in = TensorParallelEmbedding(
            prefix="gpt_neox.embed_in", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                FlashNeoXLayer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = FastLayerNorm.load(
            prefix="gpt_neox.final_layer_norm",
            weights=weights,
            eps=config.layer_norm_eps,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
    ) -> torch.Tensor:
        hidden_states = self.embed_in(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].attention.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states, _ = self.final_layer_norm(hidden_states, residual)

        return hidden_states


class FlashGPTNeoXForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.gpt_neox = FlashGPTNeoXModel(config, weights)

        self.embed_out = TensorParallelHead.load(
            config, prefix="embed_out", weights=weights
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.gpt_neox(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.embed_out(hidden_states)
        return logits
