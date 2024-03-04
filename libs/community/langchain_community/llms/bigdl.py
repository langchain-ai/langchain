import logging
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra

DEFAULT_MODEL_ID = "gpt2"

logger = logging.getLogger(__name__)


class BigdlLLM(LLM):
    """Wrapper around the BigDL-LLM Transformer-INT4 model

    Example:
        .. code-block:: python

            from langchain.llms import TransformersLLM
            llm = TransformersLLM.from_model_id(model_id="THUDM/chatglm-6b")
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name or model path to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    model: Any  #: :meta private:
    """BigDL-LLM Transformers-INT4 model."""
    tokenizer: Any  #: :meta private:
    """Huggingface tokenizer model."""
    streaming: bool = True
    """Whether to stream the results, token by token."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct object from model_id

        Args:
            model_id: Path for the huggingface repo id to be downloaded or
                      the huggingface checkpoint folder.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of TransformersLLM.
        """
        try:
            from bigdl.llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ValueError(
                "Could not import bigdl-llm or transformers. "
                "Please install it with `pip install --pre --upgrade bigdl-llm[all]`."
            )

        _model_kwargs = model_kwargs or {}

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except Exception:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, load_in_4bit=True, **_model_kwargs
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_id, load_in_4bit=True, **_model_kwargs
            )

        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @classmethod
    def from_model_id_low_bit(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct low_bit object from model_id

        Args:

            model_id: Path for the bigdl transformers low-bit model checkpoint folder.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of TransformersLLM.
        """
        try:
            from bigdl.llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ValueError(
                "Could not import bigdl-llm or transformers. "
                "Please install it with `pip install --pre --upgrade bigdl-llm[all]`"
            )

        _model_kwargs = model_kwargs or {}
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except Exception:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            model = AutoModelForCausalLM.load_low_bit(model_id, **_model_kwargs)
        except Exception:
            model = AutoModel.load_low_bit(model_id, **_model_kwargs)

        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "BigDL-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            from transformers import TextStreamer

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            if stop is not None:
                from transformers.generation.stopping_criteria import (
                    StoppingCriteriaList,
                )
                from transformers.tools.agents import StopSequenceCriteria

                # stop generation when stop words are encountered
                # TODO: stop generation when the following one is stop word
                stopping_criteria = StoppingCriteriaList(
                    [StopSequenceCriteria(stop, self.tokenizer)]
                )
            else:
                stopping_criteria = None
            output = self.model.generate(
                input_ids,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
                **kwargs,
            )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return text
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if stop is not None:
                from transformers.generation.stopping_criteria import (
                    StoppingCriteriaList,
                )
                from transformers.tools.agents import StopSequenceCriteria

                stopping_criteria = StoppingCriteriaList(
                    [StopSequenceCriteria(stop, self.tokenizer)]
                )
            else:
                stopping_criteria = None
            output = self.model.generate(
                input_ids, stopping_criteria=stopping_criteria, **kwargs
            )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)[
                len(prompt) :
            ]
            return text
