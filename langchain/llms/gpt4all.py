"""Wrapper for the GPT4All model."""
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Set

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class GPT4All(LLM):
    r"""Wrapper around GPT4All language models.

    To use, you should have the ``pyllamacpp`` python package installed, the
    pre-trained model file, and the model's config information.

    Example:
        .. code-block:: python

            from langchain.llms import GPT4All
            model = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)

            # Simplest invocation
            response = model("Once upon a time, ")
    """

    model: str
    """Path to the pre-trained GPT4All model file."""

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into. 
    If -1, the number of parts is automatically determined."""

    seed: int = Field(0, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(False, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    embedding: bool = Field(False, alias="embedding")
    """Use embedding mode only."""

    n_threads: Optional[int] = Field(4, alias="n_threads")
    """Number of threads to use."""

    n_predict: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temp: Optional[float] = 0.8
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    repeat_last_n: Optional[int] = 64
    "Last n tokens to penalize"

    repeat_penalty: Optional[float] = 1.3
    """The penalty to apply to repeated tokens."""

    n_batch: int = Field(1, alias="n_batch")
    """Batch size for prompt processing."""

    streaming: bool = False
    """Whether to stream the results or not."""

    client: Any = None  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "seed": self.seed,
            "n_predict": self.n_predict,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temp": self.temp,
        }

    @staticmethod
    def _llama_param_names() -> Set[str]:
        """Get the identifying parameters."""
        return {
            "seed",
            "n_ctx",
            "n_parts",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "embedding",
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            from pyllamacpp.model import Model as GPT4AllModel

            llama_keys = cls._llama_param_names()
            model_kwargs = {k: v for k, v in values.items() if k in llama_keys}
            values["client"] = GPT4AllModel(
                ggml_model=values["model"],
                **model_kwargs,
            )

        except ImportError:
            raise ValueError(
                "Could not import pyllamacpp python package. "
                "Please install it with `pip install pyllamacpp`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            **self._default_params,
            **{
                k: v
                for k, v in self.__dict__.items()
                if k in GPT4All._llama_param_names()
            },
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "gpt4all"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        r"""Call out to GPT4All's generate method.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "Once upon a time, "
                response = model(prompt, n_predict=55)
        """
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
            text = self.client.generate(
                prompt,
                new_text_callback=text_callback,
                **self._default_params,
            )
        else:
            text = self.client.generate(prompt, **self._default_params)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
