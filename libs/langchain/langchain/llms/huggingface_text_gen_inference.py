from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.schema.output import GenerationChunk


class HuggingFaceTextGenInference(LLM):
    """
    HuggingFace text generation API.

    It generates text from a given prompt.

    Attributes:
    - max_new_tokens: The maximum number of tokens to generate.
    - top_k: The number of top-k tokens to consider when generating text.
    - top_p: The cumulative probability threshold for generating text.
    - typical_p: The typical probability threshold for generating text.
    - temperature: The temperature to use when generating text.
    - repetition_penalty: The repetition penalty to use when generating text.
    - truncate: truncate inputs tokens to the given size
    - stop_sequences: A list of stop sequences to use when generating text.
    - seed: The seed to use when generating text.
    - inference_server_url: The URL of the inference server to use.
    - timeout: The timeout value in seconds to use while connecting to inference server.
    - server_kwargs: The keyword arguments to pass to the inference server.
    - client: The client object used to communicate with the inference server.
    - async_client: The async client object used to communicate with the server.

    Methods:
    - _call: Generates text based on a given prompt and stop sequences.
    - _acall: Async generates text based on a given prompt and stop sequences.
    - _llm_type: Returns the type of LLM.
    - _default_params: Returns the default parameters for calling text generation
     inference API.
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
                streaming = True
            )
            print(llm("What is Deep Learning?"))
            
    """

    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: float = 0.8
    repetition_penalty: Optional[float] = None
    truncate: Optional[int] = None
    stop_sequences: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    inference_server_url: str = ""
    timeout: int = 120
    server_kwargs: Dict[str, Any] = Field(default_factory=dict)
    streaming: bool = False
    client: Any
    async_client: Any

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            import text_generation

            values["client"] = text_generation.Client(
                values["inference_server_url"],
                timeout=values["timeout"],
                **values["server_kwargs"],
            )
            values["async_client"] = text_generation.AsyncClient(
                values["inference_server_url"],
                timeout=values["timeout"],
                **values["server_kwargs"],
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
        return "huggingface_textgen_inference"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling text generation inference API."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "truncate": self.truncate,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
        }

    def _invocation_params(
        self, runtime_stop: Optional[List[str]], **kwargs: Any
    ) -> Dict[str, Any]:
        params = {**self._default_params, **kwargs}
        params["stop_sequences"] = params["stop_sequences"] + (runtime_stop or [])
        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        invocation_params = self._invocation_params(stop, **kwargs)
        res = self.client.generate(prompt, **invocation_params)
        # remove stop sequences from the end of the generated text
        for stop_seq in invocation_params["stop_sequences"]:
            if stop_seq in res.generated_text:
                res.generated_text = res.generated_text[
                    : res.generated_text.index(stop_seq)
                ]
        return res.generated_text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            completion = ""
            async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        invocation_params = self._invocation_params(stop, **kwargs)
        res = await self.async_client.generate(prompt, **invocation_params)
        # remove stop sequences from the end of the generated text
        for stop_seq in invocation_params["stop_sequences"]:
            if stop_seq in res.generated_text:
                res.generated_text = res.generated_text[
                    : res.generated_text.index(stop_seq)
                ]
        return res.generated_text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)

        for res in self.client.generate_stream(prompt, **invocation_params):
            # identify stop sequence in generated text, if any
            stop_seq_found: Optional[str] = None
            for stop_seq in invocation_params["stop_sequences"]:
                if stop_seq in res.token.text:
                    stop_seq_found = stop_seq

            # identify text to yield
            text: Optional[str] = None
            if res.token.special:
                text = None
            elif stop_seq_found:
                text = res.token.text[: res.token.text.index(stop_seq_found)]
            else:
                text = res.token.text

            # yield text, if any
            if text:
                chunk = GenerationChunk(text=text)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)

            # break if stop sequence found
            if stop_seq_found:
                break

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)

        async for res in self.async_client.generate_stream(prompt, **invocation_params):
            # identify stop sequence in generated text, if any
            stop_seq_found: Optional[str] = None
            for stop_seq in invocation_params["stop_sequences"]:
                if stop_seq in res.token.text:
                    stop_seq_found = stop_seq

            # identify text to yield
            text: Optional[str] = None
            if res.token.special:
                text = None
            elif stop_seq_found:
                text = res.token.text[: res.token.text.index(stop_seq_found)]
            else:
                text = res.token.text

            # yield text, if any
            if text:
                chunk = GenerationChunk(text=text)
                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text)

            # break if stop sequence found
            if stop_seq_found:
                break
