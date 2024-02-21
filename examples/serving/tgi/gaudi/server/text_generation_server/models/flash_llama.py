import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama import LlamaTokenizer
from typing import Optional

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
    LlamaConfig,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashLlama(FlashCausalLM):
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
            raise NotImplementedError("FlashLlama is only available on GPU")

        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )

        config = LlamaConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id)

        model = FlashLlamaForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashLlama, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
