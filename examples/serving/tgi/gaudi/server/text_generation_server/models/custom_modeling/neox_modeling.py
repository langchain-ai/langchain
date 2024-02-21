# coding=utf-8
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch GPTNeoX model."""

from typing import Optional, Tuple, Union

import os
import torch
import torch.distributed
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import GPTNeoXConfig
from loguru import logger
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
        from custom_kernels import fused_attention_cuda

        CUSTOM_KERNELS_ENABLED = True
    except ImportError:
        pass

if not CUSTOM_KERNELS_ENABLED:
    logger.warning("We're not using custom kernels.")


def make_causal_mask(
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


def expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, tgt_length, src_length)


def prepare_attn_mask(
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
        combined_attention_mask = make_causal_mask(
            input_shape, device=device, past_key_values_length=past_key_values_length
        )

    # [batch_size, seq_length] -> [batch_size, tgt_length, src_length]
    expanded_attn_mask = expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask
        if combined_attention_mask is None
        else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask


class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """


class GPTNeoXAttention(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        # ??? TODO
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
        #         1, 1, max_positions, max_positions
        #     ),
        # )
        # self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            config.max_position_embeddings,
            base=config.rotary_emb_base,
        )
        self.rotary_emb.inv_freq = nn.Parameter(
            weights.get_tensor(f"{prefix}.rotary_emb.inv_freq")
        )
        self.inv_norm_factor = 1.0 / torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32)
        ).to(torch.get_default_dtype())

        if self.num_attention_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_attention_heads` must be divisible by `num_shards` "
                f"(got `num_attention_heads`: {self.num_attention_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_attention_heads = (
            self.num_attention_heads // weights.process_group.size()
        )
        self.query_key_value = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.query_key_value", weights=weights, bias=True
        )
        self.dense = TensorParallelRowLinear.load(
            config, prefix=f"{prefix}.dense", weights=weights, bias=True
        )

    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape).permute(0, 2, 1, 3)
        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query, key, value = qkv.split(self.head_size, -1)

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        key_rot = key[..., : self.rotary_ndims]

        query_rot, key_rot = self.rotary_emb(query_rot, key_rot, position_ids, seq_len)

        query[..., : self.rotary_ndims] = query_rot
        key[..., : self.rotary_ndims] = key_rot

        if CUSTOM_KERNELS_ENABLED:
            attn_output, present, attn_weights = fused_attention_cuda.forward(
                query,
                key,
                value,
                layer_past,
                attention_mask,
                head_mask,
                self.inv_norm_factor,
                self.num_attention_heads,
                use_cache,
            )
        else:
            # Cache QKV values
            if has_layer_past:
                past_key = layer_past[0]
                past_value = layer_past[1]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            present = (key, value) if use_cache else None

            # Compute attention
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

            # Reshape outputs
            attn_output = self._merge_heads(
                attn_output, self.num_attention_heads, self.head_size
            )

        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(
            tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size
        )
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        query = query.reshape(
            batch_size * num_attention_heads, query_length, attn_head_size
        )
        key = key.reshape(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            1,
            dtype=query.dtype,
            device=key.device,
        ).expand(batch_size * num_attention_heads, query_length, key_length)
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.inv_norm_factor,
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attn_scores.dtype
        if input_dtype in [torch.float16, torch.bfloat16]:
            attn_scores = attn_scores.to(torch.float)
        attn_scores = torch.where(
            attention_mask, torch.finfo(attn_scores.dtype).min, attn_scores
        )
        attn_scores = attn_scores.view(
            batch_size, num_attention_heads, query_length, key_length
        )

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        self.true_inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )
        self.register_buffer("inv_freq", self.true_inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        self.cos_cached = None
        self.sin_cached = None

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _create_cos_sin(inv_freq, max_position_embeddings, dtype, device):
        t = torch.arange(
            max_position_embeddings, device=inv_freq.device, dtype=inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(device).to(dtype), emb.sin().to(device).to(dtype)

    def forward(self, q, k, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached is None
            or self.sin_cached is None
        ):
            if seq_len > self.max_seq_len_cached:
                self.max_seq_len_cached = seq_len
            self.cos_cached, self.sin_cached = self._create_cos_sin(
                self.true_inv_freq, self.max_seq_len_cached, q.dtype, q.device
            )
        return rotary_forward(q, k, self.cos_cached, self.sin_cached, position_ids)


@torch.jit.script
def rotary_forward(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)

    chunk_size = q.shape[-1] // 2
    q1, q2 = q.split(chunk_size, -1)
    q_rotated = torch.cat((-q2, q1), dim=-1)
    k1, k2 = k.split(chunk_size, -1)
    k_rotated = torch.cat((-k2, k1), dim=-1)

    q_embed = (q * cos) + (q_rotated * sin)
    k_embed = (k * cos) + (k_rotated * sin)
    return q_embed, k_embed


class GPTNeoXMLP(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.act = (
            ACT2FN[config.hidden_act]
            if "gelu_fast" not in config.hidden_act
            else lambda x: torch.nn.functional.gelu(x, approximate="tanh")
        )

        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = TensorParallelRowLinear.load(
            config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTNeoXLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm.load(
            prefix=f"gpt_neox.layers.{layer_id}.input_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )
        self.post_attention_layernorm = nn.LayerNorm.load(
            prefix=f"gpt_neox.layers.{layer_id}.post_attention_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )
        self.attention = GPTNeoXAttention(
            config, prefix=f"gpt_neox.layers.{layer_id}.attention", weights=weights
        )
        self.mlp = GPTNeoXMLP(
            config, prefix=f"gpt_neox.layers.{layer_id}.mlp", weights=weights
        )

    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[
            0
        ]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (
                hidden_states,
            ) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class GPTNeoXModel(GPTNeoXPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.num_attention_heads = config.num_attention_heads

        self.embed_in = TensorParallelEmbedding(
            prefix="gpt_neox.embed_in", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                GPTNeoXLayer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm.load(
            prefix="gpt_neox.final_layer_norm",
            weights=weights,
            eps=config.layer_norm_eps,
        )
        self.tp_world_size = weights.process_group.size()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = inputs_embeds

        # Attention mask.
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

        causal_mask = prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        assert self.num_attention_heads % self.tp_world_size == 0
        block_size = self.num_attention_heads // self.tp_world_size
        causal_mask = torch.repeat_interleave(causal_mask, block_size, dim=0)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPTNeoxForCausalLM(GPTNeoXPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, weights):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config, weights)
        self.embed_out = TensorParallelHead.load(
            config, prefix="embed_out", weights=weights
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config.is_decoder = True
        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        input_shape = input_ids.shape

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past
