"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
import math
import os
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from einops import rearrange
from packaging import version
from text_generation_server.utils.layers import (
    TensorParallelEmbedding,
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorParallelHead,
    get_linear,
)

EPS = 1e-5


def load_col(config, prefix, weights, bias):
    assert bias == False, NotImplementedError
    assert config.quantize != "gptq", NotImplementedError
    slice_ = weights._get_slice(f"{prefix}.weight")
    rank = weights.process_group.rank()
    size = weights.process_group.size()

    h3, h = slice_.get_shape()
    block_size = h // size

    q_part = slice_[rank * block_size : (rank + 1) * block_size]
    k_part = slice_[h + rank * block_size : h + (rank + 1) * block_size]
    v_part = slice_[2 * h + rank * block_size : 2 * h + (rank + 1) * block_size]

    weight = torch.cat([q_part, k_part, v_part], dim=0)
    if weight.dtype != torch.int32:
        weight = weight.to(dtype=weights.dtype)
    weight = weight.to(device=weights.device)
    bias = None
    linear = get_linear(weight, bias, config.quantize)
    return TensorParallelColumnLinear(linear)


def _reset_is_causal(
    num_query_tokens: int, num_key_tokens: int, original_is_causal: bool
):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                "MPT does not support query and key with different number of tokens, unless number of query tokens is 1."
            )
        else:
            return False
    return original_is_causal


def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    q = rearrange(query, "b s (h d) -> b h s d", h=n_heads)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, "b s (h d) -> b h d s", h=kv_n_heads)
    v = rearrange(value, "b s (h d) -> b h s d", h=kv_n_heads)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)
    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if (
            attn_bias.size(-1) != 1
            and attn_bias.size(-1) != s_k
            or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q)
        ):
            raise RuntimeError(
                f"attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}."
            )
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn(
                "Propogating key_padding_mask to the attention module "
                + "and applying it within the attention module can cause "
                + "unneccessary computation/memory usage. Consider integrating "
                + "into attn_bias once and passing that to each attention "
                + "module instead."
            )
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), min_val
        )
    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(
            attn_weight, p=dropout_p, training=training, inplace=True
        )
    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, "b h s d -> b s (h d)")
    if needs_weights:
        return (out, attn_weight, past_key_value)
    return (out, None, past_key_value)


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(
                f"tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}."
            )
        if not tensor.is_cuda:
            raise TypeError(
                f"Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r})."
            )


def flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    try:
        from flash_attn import bert_padding, flash_attn_interface
    except:
        raise RuntimeError("Please install flash-attn==1.0.3.post0")
    check_valid_inputs(query, key, value)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if attn_bias is not None:
        raise NotImplementedError(f"attn_bias not implemented for flash attn.")
    (batch_size, seqlen) = query.shape[:2]
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1) :]
    (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(
        query, query_padding_mask
    )
    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=n_heads)
    (key_unpad, _, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(
        key, key_padding_mask
    )
    key_unpad = rearrange(
        key_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else n_heads
    )
    (value_unpad, _, _, _) = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(
        value_unpad, "nnz (h d) -> nnz h d", h=1 if multiquery else n_heads
    )
    if multiquery:
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
        value_unpad = value_unpad.expand(
            value_unpad.size(0), n_heads, value_unpad.size(-1)
        )
    dropout_p = dropout_p if training else 0.0
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    output_unpad = flash_attn_interface.flash_attn_unpadded_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=reset_is_causal,
        return_attn_probs=needs_weights,
    )
    output = bert_padding.pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices_q, batch_size, seqlen
    )
    return (output, None, past_key_value)


def triton_flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    try:
        from .flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse("2.0.0"):
            _installed = True
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            raise RuntimeError(
                "Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU and `pip install .[gpu]` if installing from llm-foundry source or `pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). Note: (1) requires you have CMake and PyTorch already installed."
            )
    check_valid_inputs(query, key, value)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if dropout_p:
        raise NotImplementedError(f"Dropout not implemented for attn_impl: triton.")
    if needs_weights:
        raise NotImplementedError(f"attn_impl: triton cannot return attn weights.")
    if key_padding_mask is not None:
        warnings.warn(
            "Propagating key_padding_mask to the attention module "
            + "and applying it within the attention module can cause "
            + "unnecessary computation/memory usage. Consider integrating "
            + "into attn_bias once and passing that to each attention "
            + "module instead."
        )
        (b_size, s_k) = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(
            ~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min
        )
    query = rearrange(query, "b s (h d) -> b s h d", h=n_heads)
    key = rearrange(key, "b s (h d) -> b s h d", h=1 if multiquery else n_heads)
    value = rearrange(value, "b s (h d) -> b s h d", h=1 if multiquery else n_heads)
    if multiquery:
        key = key.expand(*key.shape[:2], n_heads, key.size(-1))
        value = value.expand(*value.shape[:2], n_heads, value.size(-1))
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(
        query, key, value, attn_bias, reset_is_causal, softmax_scale
    )
    output = attn_output.view(*attn_output.shape[:2], -1)
    return (output, None, past_key_value)


class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        config,
        prefix,
        weights,
    ):
        super().__init__()
        attn_impl = config.attn_config["attn_impl"]
        self.attn_impl = config.attn_config["attn_impl"]
        self.clip_qkv = config.attn_config["clip_qkv"]
        self.qk_ln = config.attn_config["qk_ln"]
        self.d_model = config.d_model
        d_model = config.d_model
        self.n_heads = config.n_heads
        self.softmax_scale = config.attn_config["softmax_scale"]
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = config.attn_config["attn_pdrop"]

        if self.n_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`n_heads` must be divisible by `num_shards` (got `n_heads`: {self.n_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.n_heads = self.n_heads // weights.process_group.size()
        self.Wqkv = load_col(
            config, prefix=f"{prefix}.Wqkv", weights=weights, bias=not config.no_bias
        )
        if self.qk_ln:
            raise NotImplementedError("qk_ln is not supported")
        if self.attn_impl == "flash":
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == "triton":
            self.attn_fn = triton_flash_attn_fn
        elif self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
        else:
            raise ValueError(f"attn_impl={attn_impl!r} is an invalid setting.")
        self.out_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.out_proj",
            weights=weights,
            bias=not config.no_bias,
        )

    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        attention_mask=None,
        is_causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.chunk(3, dim=2)

        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        (context, attn_weights, past_key_value) = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
        )
        out = self.out_proj(context)
        return (out, attn_weights, past_key_value)


class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(self, config, prefix, weights):
        super().__init__()
        attn_impl = config.attn_config["attn_impl"]
        self.attn_impl = config.attn_config["attn_impl"]
        self.clip_qkv = config.attn_config["clip_qkv"]
        self.qk_ln = config.attn_config["qk_ln"]
        self.d_model = config.d_model
        d_model = config.d_model
        self.n_heads = config.n_heads
        self.softmax_scale = config.attn_config["softmax_scale"]
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = config.attn_config["attn_pdrop"]
        # self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim, device=device)
        self.Wqkv = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.Wqkv", weights=weights, bias=not config.no_bias
        )
        fuse_splits = (d_model, d_model + self.head_dim)
        if self.qk_ln:
            raise NotImplementedError("qk_ln not supported")
        if self.attn_impl == "flash":
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == "triton":
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn(
                    "While `attn_impl: triton` can be faster than `attn_impl: flash` "
                    + "it uses more memory. When training larger models this can trigger "
                    + "alloc retries which hurts performance. If encountered, we recommend "
                    + "using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`."
                )
        elif self.attn_impl == "torch":
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn(
                    "Using `attn_impl: torch`. If your model does not use `alibi` or "
                    + "`prefix_lm` we recommend using `attn_impl: flash` otherwise "
                    + "we recommend using `attn_impl: triton`."
                )
        else:
            raise ValueError(f"attn_impl={attn_impl!r} is an invalid setting.")
        self.out_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.out_proj",
            weights=weights,
            bias=not config.no_bias,
        )
        # self.out_proj._is_residual = True

    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        attention_mask=None,
        is_causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.split(
            [self.d_model, self.head_dim, self.head_dim], dim=2
        )
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        (context, attn_weights, past_key_value) = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            multiquery=True,
        )
        return (self.out_proj(context), attn_weights, past_key_value)


def attn_bias_shape(
    attn_impl, n_heads, seq_len, alibi, prefix_lm, causal, use_sequence_id
):
    if attn_impl == "flash":
        return None
    elif attn_impl in ["torch", "triton"]:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f"attn_impl={attn_impl!r} is an invalid setting.")


def build_attn_bias(
    attn_impl, attn_bias, n_heads, seq_len, causal=False, alibi=False, alibi_bias_max=8
):
    if attn_impl == "flash":
        return None
    elif attn_impl in ["torch", "triton"]:
        if alibi:
            (device, dtype) = (attn_bias.device, attn_bias.dtype)
            attn_bias = attn_bias.add(
                build_alibi_bias(
                    n_heads,
                    seq_len,
                    full=not causal,
                    alibi_bias_max=alibi_bias_max,
                    device=device,
                    dtype=dtype,
                )
            )
        return attn_bias
    else:
        raise ValueError(f"attn_impl={attn_impl!r} is an invalid setting.")


def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads, seq_len, full=False, alibi_bias_max=8, device=None, dtype=None
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(
        1, 1, 1, seq_len
    )
    if full:
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=torch.int32, device=device
        ).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)


ATTN_CLASS_REGISTRY = {
    "multihead_attention": MultiheadAttention,
    "multiquery_attention": MultiQueryAttention,
}

"""GPT Blocks used for the GPT Model."""


class MPTMLP(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        # self.up_proj = nn.Linear(d_model, expansion_ratio * d_model, device=device)
        self.up_proj = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.up_proj", weights=weights, bias=not config.no_bias
        )
        self.act = nn.GELU(approximate="none")
        # self.down_proj = nn.Linear(expansion_ratio * d_model, d_model, device=device)
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=not config.no_bias,
        )
        # self.down_proj._is_residual = True

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class MPTBlock(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.prefix = prefix
        if config.attn_config["attn_type"] != "multihead_attention":
            raise NotImplementedError(
                f"""Not implemented attn {config.attn_config["attn_type"]}"""
            )
        resid_pdrop = config.resid_pdrop
        self.norm_1 = nn.LayerNorm.load_no_bias(
            prefix=f"{prefix}.norm_1", weights=weights, eps=EPS
        )
        self.norm_2 = nn.LayerNorm.load_no_bias(
            prefix=f"{prefix}.norm_2", weights=weights, eps=EPS
        )
        self.attn = MultiheadAttention(config, prefix=f"{prefix}.attn", weights=weights)
        self.ffn = MPTMLP(config, prefix=f"{prefix}.ffn", weights=weights)
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        a = self.norm_1(x)
        (b, attn_weights, past_key_value) = self.attn(
            a,
            past_key_value=past_key_value,
            attn_bias=attn_bias,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )
        x = x + self.resid_attn_dropout(b)
        m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        return (x, attn_weights, past_key_value)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LPLayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        )
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


def rms_norm(x, weight=None, eps=1e-05):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(torch.nn.Module):
    def __init__(
        self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None
    ):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)


class LPRMSNorm(RMSNorm):
    def __init__(
        self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            weight=weight,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(dtype=x.dtype)


NORM_CLASS_REGISTRY = {
    "layernorm": torch.nn.LayerNorm,
    "low_precision_layernorm": LPLayerNorm,
    "rmsnorm": RMSNorm,
    "low_precision_rmsnorm": LPRMSNorm,
}

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class MPTPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["MPTBlock"]


class MPTModel(MPTPreTrainedModel):
    def __init__(self, config, weights):
        # config._validate_config()
        super().__init__(config)
        self.world_size = weights.process_group.size()
        self.rank = weights.process_group.rank()
        self.n_heads = config.n_heads
        self.attn_impl = config.attn_config["attn_impl"]
        self.prefix_lm = config.attn_config["prefix_lm"]
        self.attn_uses_sequence_id = config.attn_config["attn_uses_sequence_id"]
        self.alibi = config.attn_config["alibi"]
        self.alibi_bias_max = config.attn_config["alibi_bias_max"]
        if config.init_device == "mixed":
            if dist.get_local_rank() == 0:
                config.init_device = "cpu"
            else:
                config.init_device = "meta"
        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = " | ".join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(
                f"Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options})."
            )
        if config.norm_type.lower() != "low_precision_layernorm":
            raise NotImplementedError(
                f"Requested norm type ({config.norm_type}) is not implemented within this repo."
            )

        self.wte = TensorParallelEmbedding("transformer.wte", weights)
        if not self.alibi:
            # self.wpe = torch.nn.Embedding(
            #     config.max_seq_len, config.d_model, device=config.init_device
            # )
            raise RuntimeError("no alibi no supported")
        self.blocks = nn.ModuleList(
            [
                MPTBlock(config, prefix=f"transformer.blocks.{i}", weights=weights)
                for i in range(config.n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm.load_no_bias(
            prefix="transformer.norm_f", weights=weights, eps=EPS
        )
        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id,
        )
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                    if config.verbose:
                        warnings.warn(f"Removing bias ({module.bias}) from {module}.")
                    module.register_parameter("bias", None)
        if config.verbose and config.verbose > 2:
            print(self)
        if "verbose" not in self.config.init_config:
            self.config.init_config["verbose"] = self.config.verbose
        if self.config.init_config["verbose"] > 1:
            init_fn_name = self.config.init_config["name"]
            warnings.warn(f"Using {init_fn_name} initialization.")

    @torch.no_grad()
    def _attn_bias(
        self,
        device,
        dtype,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
    ):
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(
                    self.attn_bias_shape, device=device, dtype=dtype
                )
                self.attn_bias = build_attn_bias(
                    self.attn_impl,
                    self.attn_bias,
                    self.config.n_heads,
                    self.config.max_seq_len,
                    causal=self.is_causal,
                    alibi=self.alibi,
                    alibi_bias_max=self.alibi_bias_max,
                )
                assert self.n_heads % self.world_size == 0
                block_size = self.n_heads // self.world_size
                self.attn_bias = self.attn_bias[
                    :, self.rank * block_size : (self.rank + 1) * block_size
                ]
            self._attn_bias_initialized = True
        if self.attn_impl == "flash":
            return (self.attn_bias, attention_mask)
        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
        attn_bias = self.attn_bias
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)
            assert isinstance(prefix_mask, torch.Tensor)
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(
                    f"attention_mask shape={attention_mask.shape} "
                    + f"and prefix_mask shape={prefix_mask.shape} are not equal."
                )
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.view(-1, 1, 1, s_k), min_val
            )
        return (attn_bias, None)

    def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor):
        (s_k, s_q) = attn_bias.shape[-2:]
        if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
            raise ValueError(
                "attn_bias does not match the expected shape. "
                + f"The last two dimensions should both be {self.config.max_length} "
                + f"but are {s_k} and {s_q}."
            )
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}"
            )
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        causal = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)
        ).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def _apply_sequence_id(
        self, attn_bias: torch.Tensor, sequence_id: torch.LongTensor
    ):
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}"
            )
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        cannot_attend = torch.logical_not(
            torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))
        ).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()
        if not return_dict:
            raise NotImplementedError(
                "return_dict False is not implemented yet for MPT"
            )
        if output_attentions:
            if self.attn_impl != "torch":
                raise NotImplementedError(
                    "output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`."
                )
        if (
            attention_mask is not None
            and attention_mask[:, 0].sum() != attention_mask.shape[0]
            and self.training
        ):
            raise NotImplementedError(
                "MPT does not support training with left padding."
            )
        if self.prefix_lm and prefix_mask is None:
            raise ValueError(
                "prefix_mask is a required argument when MPT is configured with prefix_lm=True."
            )
        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    "sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True "
                    + "and the model is in train mode."
                )
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                warnings.warn(
                    "MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. "
                    + "This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True."
                )
        S = input_ids.size(1)
        assert (
            S <= self.config.max_seq_len
        ), f"Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}"
        tok_emb = self.wte(input_ids)
        if self.alibi:
            x = tok_emb
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        f"past_key_values must provide a past_key_value for each attention "
                        + f"layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r})."
                    )
                past_position = past_key_values[0][0].size(1)
                if self.attn_impl == "torch":
                    past_position = past_key_values[0][0].size(3)
            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f"Cannot forward input with past sequence length {past_position} and current sequence length {S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}."
                )
            pos = torch.arange(
                past_position,
                S + past_position,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
            if attention_mask is not None:
                pos = torch.clamp(
                    pos
                    - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[
                        :, past_position:
                    ],
                    min=0,
                )
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb
        (attn_bias, attention_mask) = self._attn_bias(
            device=x.device,
            dtype=torch.float32,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
        )
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for b_idx, block in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = (
                past_key_values[b_idx] if past_key_values is not None else None
            )
            (x, attn_weights, past_key_value) = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                attention_mask=attention_mask,
                is_causal=self.is_causal,
            )
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value
            if output_attentions:
                assert all_self_attns is not None
                all_self_attns = all_self_attns + (attn_weights,)
        x = self.norm_f(x)
        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (x,)
        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MPTForCausalLM(MPTPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        if not config.tie_word_embeddings:
            raise ValueError("MPTForCausalLM only supports tied word embeddings")
        self.transformer = MPTModel(config, weights)
        self.lm_head = TensorParallelHead.load(
            config, prefix="transformer.wte", weights=weights
        )
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == "inv_sqrt_d_model":
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f"Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs."
                )
            logits *= self.logit_scale
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1)
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds is not implemented for MPT yet")
        attention_mask = kwargs["attention_mask"].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError(
                "MPT does not support generation with right padding."
            )
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get("use_cache") == False:
                raise NotImplementedError(
                    "MPT with prefix_lm=True does not support use_cache=False."
                )
        else:
            prefix_mask = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_mask": prefix_mask,
            "sequence_id": sequence_id,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [
                tuple(
                    (past_state.index_select(0, beam_idx) for past_state in layer_past)
                )
            ]
        return reordered_past
