import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoTokenizer, AutoConfig
from typing import Optional, List
import json
import os

from huggingface_hub import hf_hub_download
from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_santacoder_modeling import (
    FlashSantacoderForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashSantacoderSharded(FlashCausalLM):
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
            raise NotImplementedError("FlashSantacoderSharded is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        config.quantize = quantize
        config.transpose = config.architectures[0].startswith("GPT2")

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device=device,
            dtype=dtype,
            process_group=self.process_group,
            aliases={"transformer.wte.weight": ["lm_head.weight"]},
        )
        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)

        model = FlashSantacoderForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashSantacoderSharded, self).__init__(
            model=model.to(device),
            tokenizer=tokenizer,
            num_layers=len(model.transformer.h),
            num_kv_heads=1,
            head_size=model.transformer.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
