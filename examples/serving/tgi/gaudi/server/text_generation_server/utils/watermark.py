# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
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
import os

import torch
from transformers import LogitsProcessor
from typing import List, Union

GAMMA = float(os.getenv("WATERMARK_GAMMA", 0.5))
DELTA = float(os.getenv("WATERMARK_DELTA", 2.0))


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        gamma: float = GAMMA,
        delta: float = DELTA,
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        device: str = "cpu",
    ):
        # watermarking parameters
        self.gamma = gamma
        self.delta = delta
        self.rng = torch.Generator(device="cpu")
        self.hash_key = hash_key

    def _seed_rng(self, input_ids: Union[List[int], torch.LongTensor]):
        if isinstance(input_ids, list):
            assert len(input_ids) >= 1, "requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1]
        else:
            assert len(input_ids) == 1
            input_ids = input_ids[0]
            assert input_ids.shape[-1] >= 1, "requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
        self.rng.manual_seed(self.hash_key * prev_token)

    def _get_greenlist_ids(
        self,
        input_ids: Union[List[int], torch.LongTensor],
        max_value: int,
        device: torch.device,
    ) -> List[int]:
        # seed the rng using the previous tokens/prefix
        self._seed_rng(input_ids)

        greenlist_size = int(max_value * self.gamma)
        vocab_permutation = torch.randperm(max_value, device=device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids

    @staticmethod
    def _calc_greenlist_mask(scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        green_tokens_mask[-1, greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    @staticmethod
    def _bias_greenlist_logits(
        scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: Union[List[int], torch.LongTensor], scores: torch.FloatTensor) -> torch.FloatTensor:
        greenlist_ids = self._get_greenlist_ids(input_ids, scores.shape[-1], scores.device)
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=greenlist_ids)

        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )
        return scores
