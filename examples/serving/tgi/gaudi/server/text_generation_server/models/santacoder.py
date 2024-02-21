from typing import Optional, List
import torch

from text_generation_server.models import CausalLM

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"


class SantaCoder(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model_id=model_id, revision=revision, dtype=dtype)

        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    EOD,
                    FIM_PREFIX,
                    FIM_MIDDLE,
                    FIM_SUFFIX,
                    FIM_PAD,
                ],
                "pad_token": EOD,
            }
        )

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
