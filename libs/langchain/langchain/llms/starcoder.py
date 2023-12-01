from typing import Any, Dict, List, Optional, Set
from langchain_core.pydantic_v1 import Extra
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from functools import partial
import logging

logger = logging.getLogger(__name__)


class StarCoder(LLM):
    """StarCoder Large Language Model wrapper.

    To use this class, you should have the `transformers`
    python package installed.
    """

    model_id: str = "bigcode/starcoder"
    device: Optional[str] = "cpu"
    n_predict: Optional[int] = 256
    max_tokens: int = 8196
    batch_size: int = 4
    client: Any = None  #: :meta private:

    class Config:
        extra = Extra.forbid

    @staticmethod
    def _model_param_names() -> Set[str]:
        return {
            "max_tokens",
            "n_predict",
            "batch_size",
        }

    def _default_params(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "n_predict": self.n_predict,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "bigcode/starcoder",
        device: Optional[str] = "cpu",
        max_tokens: int = 8196,
        n_predict: int = 256,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> "StarCoder":
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline as hf_pipeline,
            )
        except ImportError:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        pipeline = hf_pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "gpu" else -1 
        )


        return cls(
            client=pipeline,
            model_id=model_id,
            max_tokens=max_tokens,
            n_predict=n_predict,
            batch_size=batch_size,
            **kwargs,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        params = {**self._default_params(), **kwargs}
        generated_tokens = self.client(
            prompt, 
            max_length=params["max_tokens"], 
            num_return_sequences=params["n_predict"]
        )

        text = ""
        for token in generated_tokens:
            if text_callback:
                text_callback(token)
            text += token["generated_text"]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            **self._default_params(),
        }

    @property
    def _llm_type(self) -> str:
        return "starcoder"
