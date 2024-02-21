import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoTokenizer
from typing import Optional

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_rw_modeling import (
    RWConfig,
    FlashRWForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashRWSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashRW is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = RWConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
            aliases={
                "lm_head.weight": ["transformer.word_embeddings.weight"],
                "transformer.word_embeddings.weight": ["lm_head.weight"],
            },
        )

        config.quantize = quantize
        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)

        model = FlashRWForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashRWSharded, self).__init__(
            model=model.to(device),
            tokenizer=tokenizer,
            num_layers=len(model.transformer.h),
            num_kv_heads=model.transformer.cache_size,
            head_size=model.transformer.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
