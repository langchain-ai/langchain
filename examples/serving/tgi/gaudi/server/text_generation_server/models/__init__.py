import torch

from loguru import logger
from transformers.models.auto import modeling_auto
from transformers import AutoConfig
from typing import Optional

from text_generation_server.models.model import Model
from text_generation_server.models.causal_lm import CausalLM
from text_generation_server.models.bloom import BLOOM
from text_generation_server.models.santacoder import SantaCoder


# Disable gradients
torch.set_grad_enabled(False)


def get_model(
    model_id: str,
    revision: Optional[str],
    dtype: Optional[torch.dtype] = None,
) -> Model:
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model_type = config.model_type

    if model_type == "gpt_bigcode":
        return SantaCoder(model_id, revision, dtype)

    if model_type == "bloom":
        return BLOOM(model_id, revision, dtype)

    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(model_id, revision, dtype)

    raise ValueError(f"Unsupported model type {model_type}")
