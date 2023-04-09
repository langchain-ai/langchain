import logging
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, Extra
import torch

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import LLMResult, Generation

DEFAULT_MODEL_ID = "gpt2"

logger = logging.getLogger()


class HuggingFaceModel(LLM, BaseModel):
    model: Any
    tokenizer: Any
    model_id: str = DEFAULT_MODEL_ID
    device: int = -1
    model_kwargs: Optional[dict] = None
    generate_kwargs: Optional[dict] = None

    class Config:
        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        device: int = -1,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        try:
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            device=device,
            model_kwargs=_model_kwargs,
            generate_kwargs=generate_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_model"

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        texts = self._call(prompts, stop=stop)
        for text in texts:
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def _call(self, prompts: List[str], stop: Optional[List[str]] = None) -> str:
        _generate_kwargs = self.generate_kwargs or {}
        input_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(self.device)
        output = self.model.generate(input_ids, **_generate_kwargs)
        texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        if stop is not None:
            texts = [enforce_stop_tokens(text, stop) for text in texts]

        return texts
