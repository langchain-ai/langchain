import torch
import torch.distributed

from typing import List, Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoConfig,
)

from text_generation_server.models import Seq2SeqLM
from text_generation_server.models.custom_modeling.t5_modeling import (
    T5ForConditionalGeneration,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


class T5Sharded(Seq2SeqLM):
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

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        config.quantize = quantize

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        tokenizer.bos_token_id = config.decoder_start_token_id

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device=device,
            dtype=dtype,
            process_group=self.process_group,
            aliases={
                "shared.weight": [
                    "encoder.embed_tokens.weight",
                    "decoder.embed_tokens.weight",
                ]
            },
        )

        model = T5ForConditionalGeneration(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(Seq2SeqLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask: Optional,
        encoder_last_hidden_state: Optional,
        past_key_values: Optional = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_last_hidden_state,
            past_key_values=past_key_values,
            use_cache=True,
        )

        return (
            outputs.logits,
            outputs.encoder_last_hidden_state,
            outputs.past_key_values,
        )
