# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""

import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.distributed
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers import BloomConfig, PreTrainedModel

from text_generation_server.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    TensorParallelHead,
)

CUSTOM_KERNELS_ENABLED = False
if (
    torch.cuda.is_available()
    and not os.environ.get("DISABLE_CUSTOM_KERNELS", "False") == "True"
):
    try:
        from custom_kernels import fused_bloom_attention_cuda

        CUSTOM_KERNELS_ENABLED = True
    except ImportError:
        pass

_CHECKPOINT_FOR_DOC = "bigscience/bloom-560m"
_CONFIG_FOR_DOC = "BloomConfig"

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
]


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.ones(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    mask = mask.triu(1 + past_key_values_length)

    expanded_mask = mask.unsqueeze(0).expand(
        batch_size, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, tgt_length, src_length)


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    powers = torch.arange(
        1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32
    )
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi


# @torch.jit.script
def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


# @torch.jit.script # this is shit for unknow reasons.
def _split_heads(
    fused_qkv: torch.Tensor, num_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    storage as `fused_qkv`

    Args:
        fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    fused_qkv = fused_qkv.view(batch_size, seq_length, num_heads, 3 * head_dim)
    query_layer, key_layer, value_layer = fused_qkv.split(head_dim, dim=-1)

    query_layer = query_layer.transpose(1, 2).reshape(
        batch_size * num_heads, seq_length, head_dim
    )
    key_layer = key_layer.permute(0, 2, 3, 1).reshape(
        batch_size * num_heads, head_dim, seq_length
    )
    value_layer = value_layer.transpose(1, 2).reshape(
        batch_size * num_heads, seq_length, head_dim
    )

    return query_layer, key_layer, value_layer


# @torch.jit.script
def _merge_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Merge heads together over the last dimenstion

    Args:
        x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

    Returns:
        torch.tensor: [batch_size, seq_length, num_heads * head_dim]
    """
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    # First view to decompose the batch size
    # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
    x = x.view(batch_size, num_heads, seq_length, head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = x.permute(0, 2, 1, 3)

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return x.reshape(batch_size, seq_length, num_heads * head_dim)


class BloomAttention(nn.Module):
    def __init__(self, prefix, config: BloomConfig, weights):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.process_group = weights.process_group

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        process_group = weights.process_group
        if self.num_heads % process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {process_group.size()}"
            )
        self.num_heads = self.num_heads // process_group.size()
        self.query_key_value = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            bias=True,
        )
        self.dense = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.dense", weights=weights, bias=True
        )
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    @staticmethod
    def compute_attention(
        fused_qkv: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor],
        beta: float,
        inv_norm_factor: float,
        num_heads: int,
        use_cache: bool,
    ):
        batch_size, q_length, three_times_hidden_size = fused_qkv.shape
        head_dim = three_times_hidden_size // (3 * num_heads)
        batch_size * num_heads

        ### TODO @thomasw21: this takes quite a bit of time, how do I accelerate that?
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = _split_heads(
            fused_qkv, num_heads=num_heads, head_dim=head_dim
        )

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            past_key = past_key.view(-1, *past_key.shape[-2:])
            key_layer = torch.cat((past_key, key_layer), dim=2)
            past_value = past_value.view(-1, *past_value.shape[-2:])
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None
        ###

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=beta,
            alpha=inv_norm_factor,
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        # torch.finfo not supported by torch.jit, we temporarily remplace with `-1e34`
        attn_weights = attention_scores.masked_fill_(
            attention_mask, torch.finfo(attention_scores.dtype).min
        )
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            input_dtype
        )

        # # [batch_size, num_heads, q_length, kv_length]
        # attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs, value_layer, out=query_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = _merge_heads(
            context_layer, num_heads=num_heads, head_dim=head_dim
        )

        return context_layer, present, attention_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        batch_size, q_length, _ = fused_qkv.shape

        if layer_past is not None:
            past_key, past_value = layer_past
            layer_past = (
                past_key.view(-1, *past_key.shape[-2:]),
                past_value.view(-1, *past_value.shape[-2:]),
            )

        if CUSTOM_KERNELS_ENABLED:
            assert self.training is False, "Only foward pass was implemented"
            assert (
                attention_mask.shape[-1] < 4096
            ), "Custom kernel support only up to 4096 tokens"
            (
                context_layer,
                present,
                attention_probs,
            ) = fused_bloom_attention_cuda.forward(
                fused_qkv,
                layer_past,
                alibi,
                attention_mask,
                head_mask,
                self.beta,
                self.inv_norm_factor,
                self.num_heads,
                use_cache,
            )
        else:
            context_layer, present, attention_probs = self.compute_attention(
                fused_qkv=fused_qkv,
                layer_past=layer_past,
                alibi=alibi,
                attention_mask=attention_mask,
                head_mask=head_mask,
                beta=self.beta,
                inv_norm_factor=self.inv_norm_factor,
                num_heads=self.num_heads,
                use_cache=use_cache,
            )

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        # output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        output_tensor += residual

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Module):
    def __init__(self, prefix, config: BloomConfig, weights):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True
        )
        self.gelu_impl = torch.nn.GELU(approximate="tanh")
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[
                        :, int(i * slices) : int((i + 1) * slices)
                    ],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        # output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        intermediate_output += residual

        return intermediate_output


class BloomBlock(nn.Module):
    def __init__(self, layer_id: int, config: BloomConfig, weights):
        super().__init__()

        prefix = f"h.{layer_id}"
        self.input_layernorm = LayerNorm.load(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(
            prefix=f"{prefix}.self_attention", config=config, weights=weights
        )
        self.post_attention_layernorm = LayerNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.mlp = BloomMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class BloomPreTrainedModel(PreTrainedModel):
    config_class = BloomConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["BloomBlock"]

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_bloom_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


class BloomModel(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig, weights):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.word_embeddings = TensorParallelEmbedding(
            prefix="word_embeddings", weights=weights
        )

        self.word_embeddings_layernorm = LayerNorm.load(
            prefix="word_embeddings_layernorm",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        # Transformer blocks
        self.h = nn.ModuleList(
            [
                BloomBlock(layer_id=layer_id, config=config, weights=weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        # Final Layer Norm
        self.ln_f = LayerNorm.load(
            prefix="ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-1]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), device=hidden_states.device
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = build_alibi_tensor(attention_mask, self.num_heads)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        if hasattr(self, "tp_rank"):
            assert self.num_heads % self.tp_world_size == 0
            block_size = self.num_heads // self.tp_world_size
            alibi = alibi[
                :, self.tp_rank * block_size : (self.tp_rank + 1) * block_size
            ]
            alibi = alibi.reshape(batch_size * block_size, 1, seq_length_with_past)
            causal_mask = torch.repeat_interleave(causal_mask, block_size, dim=0)
        else:
            alibi = alibi.reshape(batch_size * self.num_heads, 1, seq_length_with_past)
            causal_mask = torch.repeat_interleave(causal_mask, self.num_heads, dim=0)

        alibi = alibi.to(hidden_states.dtype)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BloomForCausalLM(BloomPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.transformer = BloomModel(config, weights)

        self.lm_head = TensorParallelHead.load(
            config,
            prefix="word_embeddings",
            weights=weights,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        loss = None

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
