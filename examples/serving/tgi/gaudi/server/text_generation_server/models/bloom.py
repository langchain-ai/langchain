import torch

from typing import Optional, Type

from transformers import PreTrainedTokenizerBase

from text_generation_server.models import CausalLM
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.pb import generate_pb2


class BloomCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
        is_optimized_for_gaudi: bool = False,
    ) -> "CausalLMBatch":
        batch = super().from_pb(
            pb=pb,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            is_optimized_for_gaudi=is_optimized_for_gaudi,
        )
        batch.keys_head_dim_last = False
        return batch


class BLOOM(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(BLOOM, self).__init__(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return BloomCausalLMBatch
