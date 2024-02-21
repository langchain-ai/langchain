import torch
import torch.distributed

from pathlib import Path
from typing import Optional, Type
from opentelemetry import trace
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from huggingface_hub import hf_hub_download
import json

from text_generation_server.models import CausalLM
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.pb import generate_pb2
from text_generation_server.models.custom_modeling.mpt_modeling import (
    MPTForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class MPTCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "CausalLMBatch":
        batch = super().from_pb(pb=pb, tokenizer=tokenizer, dtype=dtype, device=device)
        batch.keys_head_dim_last = False
        return batch


class MPTSharded(CausalLM):
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
            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token

        # If model_id is a local path, load the file directly
        local_path = Path(model_id, "config.json")
        if local_path.exists():
            filename = str(local_path.resolve())
        else:
            filename = hf_hub_download(
                model_id, revision=revision, filename="config.json"
            )
        with open(filename, "r") as f:
            config = json.load(f)
        config = PretrainedConfig(**config)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)

        config.quantize = quantize
        model = MPTForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return MPTCausalLMBatch
