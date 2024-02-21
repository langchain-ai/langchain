# This code was adapted from https://github.com/lucidrains/flamingo-pytorch licensed under the MIT License.
#
# MIT License
#
# Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""

Generic interface to various configurations of the Perceiver Resampler, that simply takes in a series of (potentially
time-indexed) contextual embeddings, and "resamples" (compresses) them down to a pre-specified number of latents! Note
that the Perceiver in general resamples based solely off the *long-range* context; there's a nice opportunity here to
prime the Perceiver Resampler with say a single layer's worth of language embeddings (the target domain), and use that
to softly "retrieve & compress" what we need --> this would be a novel contribution we should explore.

References:
    - DeepMind's Flamingo: https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model
    - Code borrowed w/ love from: https://github.com/lucidrains/flamingo-pytorch

"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from text_generation_server.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)

EPS = 1e-5


class IdeficsPerceiverResampler(nn.Module):
    def __init__(
        self,
        prefix,
        config,
        embed_dim: int,
        depth: int,
        n_heads: int,
        head_dim: int,
        n_latents: int,
        weights,
    ) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. :param embed_dim: Dimensionality of embeddings being fed
        to the Perceiver Resampler (also dimensionality of latent embeddings *returned* by the Perceiver Resampler.
        Could be e.g., VIT embed_dim, ResNet pool dim, and so on.

        Args:
            config (`IdeficsConfig`): config object
            embed_dim (`int`): The size of each embedding vector
            depth (`int`): Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
            n_heads (`int`): Number of heads in each Transformer block (for multi-headed self-attention).
            head_dim (`int`): Dimensionality of each head projection in the Transformer block.
            n_latents (`int`):
                Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).

        """
        super().__init__()
        self.embed_dim, self.n_heads, self.head_dim, self.n_latents = (
            embed_dim,
            n_heads,
            head_dim,
            n_latents,
        )
        self.qk_layer_norms = config.perceiver_config.qk_layer_norms_perceiver

        # Create Latents for Perceiver
        self.latents = nn.Parameter(weights.get_tensor(f"{prefix}.latents"))

        self.intermediate_dim = (
            self.embed_dim * 4
            if not hasattr(config.vision_config, "embed_dim")
            else config.vision_config.embed_dim * 4
        )
        # Create Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        IdeficsPerceiverAttention(
                            prefix=f"{prefix}.blocks.{layer_id}.0",
                            config=config,
                            embed_dim=self.embed_dim,
                            n_heads=self.n_heads,
                            head_dim=self.head_dim,
                            qk_layer_norms=self.qk_layer_norms,
                            weights=weights,
                        ),
                        IdeficsMLP(
                            prefix=f"{prefix}.blocks.{layer_id}.1",
                            intermediate_size=self.intermediate_dim,
                            config=config,
                            weights=weights,
                        ),
                    ]
                )
                for layer_id in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm.load(
            prefix=f"{prefix}.layer_norm", weights=weights, eps=EPS
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Resample arbitrary length context & *compress* down to self.n_latents latent embeddings"""
        # einsum.repeat(self.latents, "seq embed -> bsz seq embed", bsz=context.shape[0])
        latents = self.latents.repeat(context.shape[0], 1, 1)

        # Feed through Perceiver Attention blocks...
        for attn, ff in self.blocks:
            latents = attn(context, latents) + latents
            latents = ff(latents) + latents

        return self.layer_norm(latents)


class IdeficsPerceiverAttention(nn.Module):
    def __init__(
        self,
        prefix,
        config,
        embed_dim: int,
        n_heads: int,
        head_dim: int,
        qk_layer_norms: bool,
        weights,
    ) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()
        self.embed_dim, self.n_heads, self.head_dim = embed_dim, n_heads, head_dim
        self.qk_layer_norms = qk_layer_norms
        # Normalization & Scaling
        self.context_layer_norm = nn.LayerNorm.load(
            prefix=f"{prefix}.context_layer_norm", weights=weights, eps=EPS
        )
        self.latents_layer_norm = nn.LayerNorm.load(
            prefix=f"{prefix}.latents_layer_norm", weights=weights, eps=EPS
        )
        if self.qk_layer_norms:
            self.q_layer_norm = nn.LayerNorm.load(
                prefix=f"{prefix}.q_layer_norm", weights=weights, eps=EPS
            )
            self.k_layer_norm = nn.LayerNorm.load(
                prefix=f"{prefix}.k_layer_norm", weights=weights, eps=EPS
            )

        self.qk_scale = self.head_dim**-0.5

        process_group = weights.process_group
        if n_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {n_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.n_heads //= weights.process_group.size()

        # Q, K, V Projection (no bias -- detail from Perceiver/Flamingo Papers).
        self.q_proj = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.q_proj", weights=weights, bias=False
        )
        self.k_proj = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.k_proj", weights=weights, bias=False
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.v_proj", weights=weights, bias=False
        )

        self.output_proj = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.output_proj", weights=weights, bias=False
        )

    def forward(self, context: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            context (`torch.Tensor`):
                Tensor of shape `[bsz, seq, embed_dim]` representing long-form context to resample.
            latents (`torch.Tensor`):
                Tensor of shape `[bsz, n_latents, embed_dim]` representing fixed length latents to compress to.

        Returns:
            `torch.Tensor`: Tensor of shape `[bsz, n_latents, embed_dim]` representing attention over latents w/ cross
            from context.
        """
        context = self.context_layer_norm(context)
        latents = self.latents_layer_norm(latents)
        batch_size, seq_length, embed_dim = context.shape[:3]

        # Query, Key, Value Projections --> Note that in Flamingo, latents are *concatenated* with context prior to attn!
        #   Note: This results in queries w/ `seq = n_latents`, and keys, values with `seq = len(context) + n_latents`
        q = self.q_proj(latents)
        k = self.k_proj(torch.cat([context, latents], dim=-2))
        v = self.v_proj(torch.cat([context, latents], dim=-2))

        # Multiheaded Self-Attention w/ stable softmax (subtract per-row max -- `amax` -- before softmax call)
        #   =>> `attn` should be a 2D matrix of shape [n_latents x (context + n_latents)]
        # einsum.rearrange(x, "bsz seq (heads embed) -> bsz heads seq embed", heads=self.n_heads)
        q, k, v = [
            x.reshape(batch_size, x.shape[1], self.n_heads, self.head_dim).transpose(
                1, 2
            )
            for x in (q, k, v)
        ]

        if self.qk_layer_norms:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)

        scores = torch.einsum("... i d, ... j d -> ... i j", q * self.qk_scale, k)
        stabilized_scores = scores - (scores.amax(dim=-1, keepdim=True).detach())
        attn = stabilized_scores.softmax(dim=-1)

        # Attend & project back to output...
        resampled = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        # einsum.rearrange(resampled, "bsz heads seq embed -> bsz seq (heads embed)", heads=self.n_heads)
        return self.output_proj(resampled.transpose(1, 2).flatten(-2))


class IdeficsMLP(nn.Module):
    def __init__(
        self,
        prefix,
        intermediate_size,
        config,
        weights,
    ):
        """Simple MLP block with intermediate_size and embedding size"""
        super().__init__()
        self.embed_dim = config.vision_config.embed_dim
        self.ln = nn.LayerNorm.load(prefix=f"{prefix}.ln", weights=weights, eps=EPS)
        self.fc = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.fc",
            weights=weights,
            bias=False,
        )
        self.act = nn.ReLU()
        self.c_proj = TensorParallelRowLinear.load(
            config=config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=False,
        )

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.ln(hidden_states)
        hidden_states = self.fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        return hidden_states
