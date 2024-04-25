from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Set

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator

from langchain_community.llms.utils import enforce_stop_tokens


class GPT4All(LLM):
    """GPT4All language models.

    To use, you should have the ``gpt4all`` python package installed, the
    pre-trained model file, and the model's config information.

    Example:
        .. code-block:: python

            from langchain_community.llms import GPT4All
            model = GPT4All(model="./models/gpt4all-model.bin", n_threads=8)

            # Simplest invocation
            response = model.invoke("Once upon a time, ")
    """

    model: str
    """Path to the pre-trained GPT4All model file."""

    backend: Optional[str] = Field(None, alias="backend")

    max_tokens: int = Field(200, alias="max_tokens")
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

    temp: Optional[float] = 0.7
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.1
    """The top-p value to use for sampling."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    repeat_last_n: Optional[int] = 64
    "Last n tokens to penalize"

    repeat_penalty: Optional[float] = 1.18
    """The penalty to apply to repeated tokens."""

    n_batch: int = Field(8, alias="n_batch")
    """Batch size for prompt processing."""

    streaming: bool = False
    """Whether to stream the results or not."""

    allow_download: bool = False
    """If model does not exist in ~/.cache/gpt4all/, download it."""

    device: Optional[str] = Field("cpu", alias="device")
    """Device name: cpu, gpu, nvidia, intel, amd or DeviceName."""

    client: Any = None  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @staticmethod
    def _model_param_names() -> Set[str]:
        return {
            "max_tokens",
            "n_predict",
            "top_k",
            "top_p",
            "temp",
            "n_batch",
            "repeat_penalty",
            "repeat_last_n",
            "streaming",
        }

    def _default_params(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "n_predict": self.n_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temp": self.temp,
            "n_batch": self.n_batch,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "streaming": self.streaming,
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            from gpt4all import GPT4All as GPT4AllModel
        except ImportError:
            raise ImportError(
                "Could not import gpt4all python package. "
                "Please install it with `pip install gpt4all`."
            )

        full_path = values["model"]
        model_path, delimiter, model_name = full_path.rpartition("/")
        model_path += delimiter

        values["client"] = GPT4AllModel(
            model_name,
            model_path=model_path or None,
            model_type=values["backend"],
            allow_download=values["allow_download"],
            device=values["device"],
        )
        if values["n_threads"] is not None:
            # set n_threads
            values["client"].model.set_thread_count(values["n_threads"])

        try:
            values["backend"] = values["client"].model_type
        except AttributeError:
            # The below is for compatibility with GPT4All Python bindings <= 0.2.3.
            values["backend"] = values["client"].model.model_type

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            **self._default_params(),
            **{
                k: v for k, v in self.__dict__.items() if k in self._model_param_names()
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
        **kwargs: Any,
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
                response = model.invoke(prompt, n_predict=55)
        """
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        params = {**self._default_params(), **kwargs}
        for token in self.client.generate(prompt, **params):
            if text_callback:
                text_callback(token)
            text += token
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
