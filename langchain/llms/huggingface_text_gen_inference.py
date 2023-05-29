"""Wrapper around Huggingface text generation inference API."""
from functools import partial
from typing import Any, Dict, List, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class HuggingFaceTextGenInference(LLM):
    """
    HuggingFace text generation inference API.

    This class is a wrapper around the HuggingFace text generation inference API.
    It is used to generate text from a given prompt.

    Attributes:
    - max_new_tokens: The maximum number of tokens to generate.
    - top_k: The number of top-k tokens to consider when generating text.
    - top_p: The cumulative probability threshold for generating text.
    - typical_p: The typical probability threshold for generating text.
    - temperature: The temperature to use when generating text.
    - repetition_penalty: The repetition penalty to use when generating text.
    - stop_sequences: A list of stop sequences to use when generating text.
    - seed: The seed to use when generating text.
    - inference_server_url: The URL of the inference server to use.
    - timeout: The timeout value in seconds to use while connecting to inference server.
    - client: The client object used to communicate with the inference server.

    Methods:
    - _call: Generates text based on a given prompt and stop sequences.
    - _llm_type: Returns the type of LLM.
    """

    """
    Example:
        .. code-block:: python

            # Basic Example (no streaming)
            llm = HuggingFaceTextGenInference(
                inference_server_url = "http://localhost:8010/",
                max_new_tokens = 512,
                top_k = 10,
                top_p = 0.95,
                typical_p = 0.95,
                temperature = 0.01,
                repetition_penalty = 1.03,
            )
            print(llm("What is Deep Learning?"))
            
            # Streaming response example
            from langchain.callbacks import streaming_stdout
            
            callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
            llm = HuggingFaceTextGenInference(
                inference_server_url = "http://localhost:8010/",
                max_new_tokens = 512,
                top_k = 10,
                top_p = 0.95,
                typical_p = 0.95,
                temperature = 0.01,
                repetition_penalty = 1.03,
                callbacks = callbacks,
                stream = True
            )
            print(llm("What is Deep Learning?"))
            
    """

    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: float = 0.8
    repetition_penalty: Optional[float] = None
    stop_sequences: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    inference_server_url: str = ""
    timeout: int = 120
    stream: bool = False
    client: Any

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            import text_generation

            values["client"] = text_generation.Client(
                values["inference_server_url"], timeout=values["timeout"]
            )
        except ImportError:
            raise ImportError(
                "Could not import text_generation python package. "
                "Please install it with `pip install text_generation`."
            )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "hf_textgen_inference"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is None:
            stop = self.stop_sequences
        else:
            stop += self.stop_sequences

        if not self.stream:
            res = self.client.generate(
                prompt,
                stop_sequences=stop,
                max_new_tokens=self.max_new_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                typical_p=self.typical_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                seed=self.seed,
            )
            # remove stop sequences from the end of the generated text
            for stop_seq in stop:
                if stop_seq in res.generated_text:
                    res.generated_text = res.generated_text[
                        : res.generated_text.index(stop_seq)
                    ]
            text = res.generated_text
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )
            params = {
                "stop_sequences": stop,
                "max_new_tokens": self.max_new_tokens,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "typical_p": self.typical_p,
                "temperature": self.temperature,
                "repetition_penalty": self.repetition_penalty,
                "seed": self.seed,
            }
            text = ""
            for res in self.client.generate_stream(prompt, **params):
                token = res.token
                is_stop = False
                for stop_seq in stop:
                    if stop_seq in token.text:
                        is_stop = True
                        break
                if is_stop:
                    break
                if not token.special:
                    if text_callback:
                        text_callback(token.text)
        return text
