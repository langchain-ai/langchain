import logging
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

DEFAULT_MODEL_ID = "gpt2"


logger = logging.getLogger(__name__)


class IpexLLM(LLM):
    """IpexLLM model.

    Example:
        .. code-block:: python

            from langchain_community.llms import IpexLLM
            llm = IpexLLM.from_model_id(model_id="THUDM/chatglm-6b")
    """

    model_id: str = DEFAULT_MODEL_ID
    """Model name or model path to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments passed to the model."""
    model: Any  #: :meta private:
    """IpexLLM model."""
    tokenizer: Any  #: :meta private:
    """Huggingface tokenizer model."""
    streaming: bool = True
    """Whether to stream the results, token by token."""

    class Config:
        extra = "forbid"

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        *,
        tokenizer_id: Optional[str] = None,
        load_in_4bit: bool = True,
        load_in_low_bit: Optional[str] = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct object from model_id

        Args:
            model_id: Path for the huggingface repo id to be downloaded or
                      the huggingface checkpoint folder.
            tokenizer_id: Path for the huggingface repo id to be downloaded or
                      the huggingface checkpoint folder which contains the tokenizer.
            load_in_4bit: "Whether to load model in 4bit.
                      Unused if `load_in_low_bit` is not None.
            load_in_low_bit: Which low bit precisions to use when loading model.
                      Example values: 'sym_int4', 'asym_int4', 'fp4', 'nf4', 'fp8', etc.
                      Overrides `load_in_4bit` if specified.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of IpexLLM.

        """

        return cls._load_model(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            low_bit_model=False,
            load_in_4bit=load_in_4bit,
            load_in_low_bit=load_in_low_bit,
            model_kwargs=model_kwargs,
            kwargs=kwargs,
        )

    @classmethod
    def from_model_id_low_bit(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        *,
        tokenizer_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct low_bit object from model_id

        Args:

            model_id: Path for the ipex-llm transformers low-bit model folder.
            tokenizer_id: Path for the huggingface repo id or local model folder
                      which contains the tokenizer.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of IpexLLM.
        """

        return cls._load_model(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            low_bit_model=True,
            load_in_4bit=False,  # not used for low-bit model
            load_in_low_bit=None,  # not used for low-bit model
            model_kwargs=model_kwargs,
            kwargs=kwargs,
        )

    @classmethod
    def _load_model(
        cls,
        model_id: str,
        tokenizer_id: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_low_bit: Optional[str] = None,
        low_bit_model: bool = False,
        model_kwargs: Optional[dict] = None,
        kwargs: Optional[dict] = None,
    ) -> Any:
        try:
            from ipex_llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ImportError(
                "Could not import ipex-llm. "
                "Please install `ipex-llm` properly following installation guides: "
                "https://github.com/intel-analytics/ipex-llm?tab=readme-ov-file#install-ipex-llm."
            )

        _model_kwargs = model_kwargs or {}
        kwargs = kwargs or {}

        _tokenizer_id = tokenizer_id or model_id
        # Set "cpu" as default device
        if "device" not in _model_kwargs:
            _model_kwargs["device"] = "cpu"

        if _model_kwargs["device"] not in ["cpu", "xpu"]:
            raise ValueError(
                "IpexLLMBgeEmbeddings currently only supports device to be "
                f"'cpu' or 'xpu', but you have: {_model_kwargs['device']}."
            )
        device = _model_kwargs.pop("device")

        try:
            tokenizer = AutoTokenizer.from_pretrained(_tokenizer_id, **_model_kwargs)
        except Exception:
            tokenizer = LlamaTokenizer.from_pretrained(_tokenizer_id, **_model_kwargs)

        # restore model_kwargs
        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        # load model with AutoModelForCausalLM and falls back to AutoModel on failure.
        load_kwargs = {
            "use_cache": True,
            "trust_remote_code": True,
        }

        if not low_bit_model:
            if load_in_low_bit is not None:
                load_function_name = "from_pretrained"
                load_kwargs["load_in_low_bit"] = load_in_low_bit  # type: ignore
            else:
                load_function_name = "from_pretrained"
                load_kwargs["load_in_4bit"] = load_in_4bit
        else:
            load_function_name = "load_low_bit"

        try:
            # Attempt to load with AutoModelForCausalLM
            model = cls._load_model_general(
                AutoModelForCausalLM,
                load_function_name=load_function_name,
                model_id=model_id,
                load_kwargs=load_kwargs,
                model_kwargs=_model_kwargs,
            )
        except Exception:
            # Fallback to AutoModel if there's an exception
            model = cls._load_model_general(
                AutoModel,
                load_function_name=load_function_name,
                model_id=model_id,
                load_kwargs=load_kwargs,
                model_kwargs=_model_kwargs,
            )

        model.to(device)

        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @staticmethod
    def _load_model_general(
        model_class: Any,
        load_function_name: str,
        model_id: str,
        load_kwargs: dict,
        model_kwargs: dict,
    ) -> Any:
        """General function to attempt to load a model."""
        try:
            load_function = getattr(model_class, load_function_name)
            return load_function(model_id, **{**load_kwargs, **model_kwargs})
        except Exception as e:
            logger.error(
                f"Failed to load model using "
                f"{model_class.__name__}.{load_function_name}: {e}"
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
        return "ipex-llm"

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
            input_ids = input_ids.to(self.model.device)
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
            input_ids = input_ids.to(self.model.device)
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
