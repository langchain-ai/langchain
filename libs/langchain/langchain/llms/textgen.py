import json
import logging
from typing import Any, Dict, Iterator, List, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Field
from langchain.schema.output import GenerationChunk

logger = logging.getLogger(__name__)


class TextGen(LLM):
    """text-generation-webui models.

    To use, you should have the text-generation-webui installed, a model loaded,
    and --api added as a command-line option.

    Suggested installation, use one-click installer for your OS:
    https://github.com/oobabooga/text-generation-webui#one-click-installers

    Parameters below taken from text-generation-webui api example:
    https://github.com/oobabooga/text-generation-webui/blob/main/api-examples/api-example.py

    Example:
        .. code-block:: python

            from langchain.llms import TextGen
            llm = TextGen(model_url="http://localhost:8500")
    """

    model_url: str
    """The full URL to the textgen webui including http[s]://host:port """

    preset: Optional[str] = None
    """The preset to use in the textgen webui """

    max_new_tokens: Optional[int] = 250
    """The maximum number of tokens to generate."""

    do_sample: bool = Field(True, alias="do_sample")
    """Do sample"""

    temperature: Optional[float] = 1.3
    """Primary factor to control randomness of outputs. 0 = deterministic
    (only the most likely token is used). Higher value = more randomness."""

    top_p: Optional[float] = 0.1
    """If not set to 1, select tokens with probabilities adding up to less than this
    number. Higher value = higher range of possible random results."""

    typical_p: Optional[float] = 1
    """If not set to 1, select only tokens that are at least this much more likely to
    appear than random tokens, given the prior text."""

    epsilon_cutoff: Optional[float] = 0  # In units of 1e-4
    """Epsilon cutoff"""

    eta_cutoff: Optional[float] = 0  # In units of 1e-4
    """ETA cutoff"""

    repetition_penalty: Optional[float] = 1.18
    """Exponential penalty factor for repeating prior tokens. 1 means no penalty,
    higher value = less repetition, lower value = more repetition."""

    top_k: Optional[float] = 40
    """Similar to top_p, but select instead only the top_k most likely tokens.
    Higher value = higher range of possible random results."""

    min_length: Optional[int] = 0
    """Minimum generation length in tokens."""

    no_repeat_ngram_size: Optional[int] = 0
    """If not set to 0, specifies the length of token sets that are completely blocked
    from repeating at all. Higher values = blocks larger phrases,
    lower values = blocks words or letters from repeating.
    Only 0 or high values are a good idea in most cases."""

    num_beams: Optional[int] = 1
    """Number of beams"""

    penalty_alpha: Optional[float] = 0
    """Penalty Alpha"""

    length_penalty: Optional[float] = 1
    """Length Penalty"""

    early_stopping: bool = Field(False, alias="early_stopping")
    """Early stopping"""

    seed: int = Field(-1, alias="seed")
    """Seed (-1 for random)"""

    add_bos_token: bool = Field(True, alias="add_bos_token")
    """Add the bos_token to the beginning of prompts.
    Disabling this can make the replies more creative."""

    truncation_length: Optional[int] = 2048
    """Truncate the prompt up to this length. The leftmost tokens are removed if
    the prompt exceeds this length. Most models require this to be at most 2048."""

    ban_eos_token: bool = Field(False, alias="ban_eos_token")
    """Ban the eos_token. Forces the model to never end the generation prematurely."""

    skip_special_tokens: bool = Field(True, alias="skip_special_tokens")
    """Skip special tokens. Some specific models need this unset."""

    stopping_strings: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    streaming: bool = False
    """Whether to stream the results, token by token."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling textgen."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "epsilon_cutoff": self.epsilon_cutoff,
            "eta_cutoff": self.eta_cutoff,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "min_length": self.min_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "penalty_alpha": self.penalty_alpha,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "seed": self.seed,
            "add_bos_token": self.add_bos_token,
            "truncation_length": self.truncation_length,
            "ban_eos_token": self.ban_eos_token,
            "skip_special_tokens": self.skip_special_tokens,
            "stopping_strings": self.stopping_strings,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_url": self.model_url}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "textgen"

    def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters in format needed by textgen.

        Args:
            stop (Optional[List[str]]): List of stop sequences for textgen.

        Returns:
            Dictionary containing the combined parameters.
        """

        # Raise error if stop sequences are in both input and default params
        # if self.stop and stop is not None:
        if self.stopping_strings and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")

        if self.preset is None:
            params = self._default_params
        else:
            params = {"preset": self.preset}

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stopping_strings or stop or []

        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the textgen web API and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                from langchain.llms import TextGen
                llm = TextGen(model_url="http://localhost:5000")
                llm("Write a story about llamas.")
        """
        if self.streaming:
            combined_text_output = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                combined_text_output += chunk.text
            print(prompt + combined_text_output)
            result = combined_text_output

        else:
            url = f"{self.model_url}/api/v1/generate"
            params = self._get_parameters(stop)
            request = params.copy()
            request["prompt"] = prompt
            response = requests.post(url, json=request)

            if response.status_code == 200:
                result = response.json()["results"][0]["text"]
                print(prompt + result)
            else:
                print(f"ERROR: response: {response}")
                result = ""

        return result

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See text-generation-webui docs and below for more.

        Example:
            .. code-block:: python

                from langchain.llms import TextGen
                llm = TextGen(
                    model_url = "ws://localhost:5005"
                    streaming=True
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                        stop=["'","\n"]):
                    print(chunk, end='', flush=True)

        """
        try:
            import websocket
        except ImportError:
            raise ImportError(
                "The `websocket-client` package is required for streaming."
            )

        params = {**self._get_parameters(stop), **kwargs}

        url = f"{self.model_url}/api/v1/stream"

        request = params.copy()
        request["prompt"] = prompt

        websocket_client = websocket.WebSocket()

        websocket_client.connect(url)

        websocket_client.send(json.dumps(request))

        while True:
            result = websocket_client.recv()
            result = json.loads(result)

            if result["event"] == "text_stream":
                chunk = GenerationChunk(
                    text=result["text"],
                    generation_info=None,
                )
                yield chunk
            elif result["event"] == "stream_end":
                websocket_client.close()
                return

            if run_manager:
                run_manager.on_llm_new_token(token=chunk.text)
