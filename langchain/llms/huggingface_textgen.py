"""Wrapper for models served by HuggingFace text-generation API."""
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class HuggingFaceTextgen(LLM):
    r"""Wrapper around HuggingFace text-generation inference API for language models.

    To use, you should have the ``text-generation`` python package installed, and
    a running inference server started up and accessible on the local network.

    Example:
        .. code-block:: python

            # Both of these examples assume you already have a local LLM server
            # running. See the HuggingFace text-generation documentation for more
            # information on how to do this.
            from langchain.llms import HuggingFaceTextgen

            prompt = "What is Deep Learning?"
            host = "localhost"
            port = 8080

            # Basic example (no streaming)
            llm = HuggingFaceTextgen(host=host, port=port)
            print(llm(prompt))

            # Streaming response example
            from langchain.callbacks import streaming_stdout

            callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
            llm = HuggingFaceTextgen(
                host=host,
                port=port,
                callbacks=callbacks,
                stream=True
            )
            print(llm(prompt))

    """

    host: str = "localhost"
    """Hostname for the huggingface-text-generation API endpoint 
    (default=localhost)."""

    port: int = 8080
    """Port for the huggingface-text-generation API endpoint (default=8080)."""

    timeout: Optional[int] = 10
    """Timeout for the huggingface-text-generation API endpoint (default=10)."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """The number of highest probability vocabulary tokens to keep for 
    top-k-filtering."""

    top_p: Optional[float] = None
    """If set to < 1, only the smallest set of most probable tokens with 
    probabilities that add up to `top_p` or higher are kept for generation."""

    truncate: Optional[int] = None
    """Truncate inputs tokens to the given size."""

    typical_p: Optional[float] = None
    """Typical Decoding mass."""

    do_sample: Optional[bool] = False
    """Activate logits sampling."""

    max_new_tokens: Optional[int] = 200
    """The maximum number of tokens to generate."""

    repetition_penalty: Optional[float] = None
    """The parameter for repetition penalty. 1.0 means no penalty."""

    return_full_text: Optional[bool] = False
    """Whether to prepend the prompt to the generated text."""

    seed: Optional[int] = None
    """Seed. If -1, a random seed is used."""

    best_of: Optional[int] = 1
    """Generate best_of sequences and return the one if the highest 
    token logprobs."""

    stop_sequences: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    stream: Optional[bool] = False
    """Whether to stream the results or not."""

    watermark: Optional[bool] = False
    """Watermarking with 
    [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)"""

    client: Any = None  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        # TODO: Let user specify these in class instantiation
        return {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "best_of": self.best_of,
            "repetition_penalty": self.repetition_penalty,
            "return_full_text": self.return_full_text,
            "seed": self.seed,
            "stop_sequences": self.stop_sequences,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "truncate": self.truncate,
            "typical_p": self.typical_p,
            "watermark": self.watermark,
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            import text_generation

            values["client"] = text_generation.Client(
                base_url=f"http://{values['host']}:{values['port']}",
                timeout=values["timeout"],
            )

        except ImportError:
            raise ValueError(
                "Could not import text_generation python package. "
                "Please install it with `pip install text-generation`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "host": self.host,
            "port": self.port,
            **self._default_params,
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "hf-text-generation"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call out to a local HuggingFace text-generation API endpoints.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        """
        if not self.stream:
            params = self._default_params
            text = self.client.generate(prompt, **params).generated_text
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )
            text = ""
            params = self._default_params
            # Remove the best_of parameter that the stream endpoint doesn't support
            params.pop("best_of")
            for response in self.client.generate_stream(prompt, **params):
                if not response.token.special:
                    token = response.token.text
                    if text_callback:
                        text_callback(token)

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
