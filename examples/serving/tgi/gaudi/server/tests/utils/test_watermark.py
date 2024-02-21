# test_watermark_logits_processor.py

import os
import numpy as np
import torch
from text_generation_server.utils.watermark import WatermarkLogitsProcessor


GAMMA = os.getenv("WATERMARK_GAMMA", 0.5)
DELTA = os.getenv("WATERMARK_DELTA", 2.0)


def test_seed_rng():
    input_ids = [101, 2036, 3731, 102, 2003, 103]
    processor = WatermarkLogitsProcessor()
    processor._seed_rng(input_ids)
    assert isinstance(processor.rng, torch.Generator)


def test_get_greenlist_ids():
    input_ids = [101, 2036, 3731, 102, 2003, 103]
    processor = WatermarkLogitsProcessor()
    result = processor._get_greenlist_ids(input_ids, 10, torch.device("cpu"))
    assert max(result) <= 10
    assert len(result) == int(10 * 0.5)


def test_calc_greenlist_mask():
    processor = WatermarkLogitsProcessor()
    scores = torch.tensor([[0.5, 0.3, 0.2, 0.8], [0.1, 0.2, 0.7, 0.9]])
    greenlist_token_ids = torch.tensor([2, 3])
    result = processor._calc_greenlist_mask(scores, greenlist_token_ids)
    assert result.tolist() == [[False, False, False, False], [False, False, True, True]]
    assert result.shape == scores.shape


def test_bias_greenlist_logits():
    processor = WatermarkLogitsProcessor()
    scores = torch.tensor([[0.5, 0.3, 0.2, 0.8], [0.1, 0.2, 0.7, 0.9]])
    green_tokens_mask = torch.tensor(
        [[False, False, True, True], [False, False, False, True]]
    )
    greenlist_bias = 2.0
    result = processor._bias_greenlist_logits(scores, green_tokens_mask, greenlist_bias)
    assert np.allclose(result.tolist(), [[0.5, 0.3, 2.2, 2.8], [0.1, 0.2, 0.7, 2.9]])
    assert result.shape == scores.shape


def test_call():
    input_ids = [101, 2036, 3731, 102, 2003, 103]
    processor = WatermarkLogitsProcessor()
    scores = torch.tensor([[0.5, 0.3, 0.2, 0.8], [0.1, 0.2, 0.7, 0.9]])
    result = processor(input_ids, scores)
    assert result.shape == scores.shape
