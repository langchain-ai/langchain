import logging
from typing import Any, Optional

from langchain_core.language_models.llms import LLM

from langchain_community.llms.ipex_llm import IpexLLM

logger = logging.getLogger(__name__)


class BigdlLLM(IpexLLM):
    """Wrapper around the BigdlLLM model

    Example:
        .. code-block:: python

            from langchain_community.llms import BigdlLLM
            llm = BigdlLLM.from_model_id(model_id="THUDM/chatglm-6b")
    """

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
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of BigdlLLM.
        """
        logger.warning("BigdlLLM was deprecated. Please use IpexLLM instead.")

        try:
            from bigdl.llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ImportError(
                "Could not import bigdl-llm or transformers. "
                "Please install it with `pip install --pre --upgrade bigdl-llm[all]`."
            )

        if load_in_low_bit is not None:
            logger.warning(
                """`load_in_low_bit` option is not supported in BigdlLLM and 
                is ignored. For more data types support with `load_in_low_bit`, 
                use IpexLLM instead."""
            )

        if not load_in_4bit:
            raise ValueError(
                "BigdlLLM only supports loading in 4-bit mode, "
                "i.e. load_in_4bit = True. "
                "Please install it with `pip install --pre --upgrade bigdl-llm[all]`."
            )

        _model_kwargs = model_kwargs or {}
        _tokenizer_id = tokenizer_id or model_id

        try:
            tokenizer = AutoTokenizer.from_pretrained(_tokenizer_id, **_model_kwargs)
        except Exception:
            tokenizer = LlamaTokenizer.from_pretrained(_tokenizer_id, **_model_kwargs)

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
        *,
        tokenizer_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct low_bit object from model_id

        Args:

            model_id: Path for the bigdl-llm transformers low-bit model folder.
            tokenizer_id: Path for the huggingface repo id or local model folder
                      which contains the tokenizer.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of BigdlLLM.
        """

        logger.warning("BigdlLLM was deprecated. Please use IpexLLM instead.")

        try:
            from bigdl.llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ImportError(
                "Could not import bigdl-llm or transformers. "
                "Please install it with `pip install --pre --upgrade bigdl-llm[all]`."
            )

        _model_kwargs = model_kwargs or {}
        _tokenizer_id = tokenizer_id or model_id

        try:
            tokenizer = AutoTokenizer.from_pretrained(_tokenizer_id, **_model_kwargs)
        except Exception:
            tokenizer = LlamaTokenizer.from_pretrained(_tokenizer_id, **_model_kwargs)

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
    def _llm_type(self) -> str:
        return "bigdl-llm"
