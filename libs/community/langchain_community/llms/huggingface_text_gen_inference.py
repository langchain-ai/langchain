import logging
from abc import ABC, abstractmethod
from collections import UserString, deque
from typing import Any, AsyncIterator, Deque, Dict, Iterator, List, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_pydantic_field_names

logger = logging.getLogger(__name__)


class HuggingFaceTextGenInference(LLM):
    """
    HuggingFace text generation API.

    To use, you should have the `text-generation` python package installed and
    a text-generation server running.

    Example:
        .. code-block:: python

            # Basic Example (no streaming)
            llm = HuggingFaceTextGenInference(
                inference_server_url="http://localhost:8010/",
                max_new_tokens=512,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
            )
            print(llm("What is Deep Learning?"))

            # Streaming response example
            from langchain_community.callbacks import streaming_stdout

            callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
            llm = HuggingFaceTextGenInference(
                inference_server_url="http://localhost:8010/",
                max_new_tokens=512,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                callbacks=callbacks,
                streaming=True
            )
            print(llm("What is Deep Learning?"))

    """

    max_new_tokens: int = 512
    """Maximum number of generated tokens"""
    top_k: Optional[int] = None
    """The number of highest probability vocabulary tokens to keep for
    top-k-filtering."""
    top_p: Optional[float] = 0.95
    """If set to < 1, only the smallest set of most probable tokens with probabilities
    that add up to `top_p` or higher are kept for generation."""
    typical_p: Optional[float] = 0.95
    """Typical Decoding mass. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information."""
    temperature: Optional[float] = 0.8
    """The value used to module the logits distribution."""
    repetition_penalty: Optional[float] = None
    """The parameter for repetition penalty. 1.0 means no penalty.
    See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details."""
    return_full_text: bool = False
    """Whether to prepend the prompt to the generated text"""
    truncate: Optional[int] = None
    """Truncate inputs tokens to the given size"""
    stop_sequences: List[str] = Field(default_factory=list)
    """Stop generating tokens if a member of `stop_sequences` is generated"""
    seed: Optional[int] = None
    """Random sampling seed"""
    inference_server_url: str = ""
    """text-generation-inference instance base url"""
    timeout: int = 120
    """Timeout in seconds"""
    streaming: bool = False
    """Whether to generate a stream of tokens asynchronously"""
    do_sample: bool = False
    """Activate logits sampling"""
    watermark: bool = False
    """Watermarking with [A Watermark for Large Language Models]
    (https://arxiv.org/abs/2301.10226)"""
    server_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any text-generation-inference server parameters not explicitly specified"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `call` not explicitly specified"""
    client: Any
    async_client: Any

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

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
            "return_full_text": self.return_full_text,
            "truncate": self.truncate,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
            "do_sample": self.do_sample,
            "watermark": self.watermark,
            **self.model_kwargs,
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
        streamer = _get_streamer(invocation_params["stop_sequences"])
        for res in self.client.generate_stream(prompt, **invocation_params):
            text = streamer(res.token.text, res.token.special)
            if text:
                yield from self._yield_chunk(text, run_manager)
            if streamer.end_reached():
                break
        if text := streamer.return_remaining_text():
            yield from self._yield_chunk(text, run_manager)

    def _yield_chunk(
        self, text: str, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None
    ) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text=text)
        if run_manager:
            run_manager.on_llm_new_token(text)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)
        streamer = _get_streamer(invocation_params["stop_sequences"])
        async for res in self.async_client.generate_stream(prompt, **invocation_params):
            if text := streamer(res.token.text, res.token.special):
                async for token in self._ayield_chunk(text, run_manager):
                    yield token
            if streamer.end_reached():
                break
        if text := streamer.return_remaining_text():
            async for token in self._ayield_chunk(text, run_manager):
                yield token

    async def _ayield_chunk(
        self, text: str, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None
    ) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text=text)
        if run_manager:
            await run_manager.on_llm_new_token(text)


###
# Streamers
###
class Streamer(ABC):
    """A class that tracks the state of a stream of tokens and yields text"""

    @abstractmethod
    def __call__(self, token: str, is_special_token: bool) -> Optional[str]:
        """Returns the text to yield, if any, given the token."""

    @abstractmethod
    def end_reached(self) -> bool:
        """whether or not we have hit the end of the stream"""

    @abstractmethod
    def return_remaining_text(self) -> Optional[str]:
        """Returns the remaining text to yield, if any"""


class PassThroughStreamer(Streamer):
    """A streamer that yields all tokens"""

    def __call__(self, token: str, is_special_token: bool) -> Optional[str]:
        return None if is_special_token else token

    def end_reached(self) -> bool:
        return False

    def return_remaining_text(self) -> Optional[str]:
        return None


class StopSequenceAwareStreamer(Streamer):
    """Tracks whether a stop sequence has been hit. If so, stops streaming."""

    def __init__(self, stop_sequences: List[str]) -> None:
        assert stop_sequences, "stop_sequences must be non-empty"
        self._stop_seqs_by_len = sorted(stop_sequences, key=len, reverse=True)
        self._buffer = TokenStreamBuffer()
        self._end_reached = False

    def __call__(self, token: str, is_special_token: bool) -> Optional[str]:
        self._buffer.extend(token, is_special_token)
        for stop_seq in self._stop_seqs_by_len:  # check for stop seq in buffer
            if 0 < (stop_seq_idx := self._buffer.find(stop_seq)):
                self._end_reached = True
                return self._buffer.stream_chars(stop_seq_idx)  # stream up to stop seq
        for stop_seq in self._stop_seqs_by_len:  # check potentially upcoming stop seqs
            if has_overlapping_chars(self._buffer, stop_seq):
                return None  # wait for next token to see if stop seq gets hit
        return self._buffer.stream_chars()

    def end_reached(self) -> bool:
        return self._end_reached

    def return_remaining_text(self) -> Optional[str]:
        return self._buffer.stream_chars() or None


def _get_streamer(stop_sequences: List[str]) -> "Streamer":
    if not stop_sequences:
        return PassThroughStreamer()
    return StopSequenceAwareStreamer(stop_sequences)


###
# TokenStreamBuffer
###
class TokenStreamBuffer(UserString):
    """A buffer that stores the last n (:=max_size) characters added to it."""

    def __init__(self, max_size: Optional[int] = None):
        super().__init__("")
        self._string_buffer: Deque[str] = deque(maxlen=max_size)
        self._stream_char_mask: Deque[bool] = deque(maxlen=max_size)

    def extend(self, token: str, is_special: bool) -> None:
        """Adds new_chars to the end of the buffer"""
        for char in token:
            self._string_buffer.append(char)  # append all chars to buffer
            self._stream_char_mask.append(not is_special)  # don't stream special tokens
        self._update_data()

    def stream_chars(self, k: Optional[int] = None) -> str:
        """Pops k characters from the beginning of the stringbuffer"""
        chars_to_stream: List[str] = []
        for _ in range(k or len(self)):
            char = self._string_buffer.popleft()
            if self._stream_char_mask.popleft():
                chars_to_stream.append(char)
        self._update_data()
        return "".join(chars_to_stream)

    def _update_data(self) -> None:
        self.data = "".join(self._string_buffer)


def has_overlapping_chars(left_str: Union[str, UserString], right_str: str) -> bool:
    """Returns the number of overlapping characters between the end
    of left_string and the beginning of right_string
    """
    idxs = reversed(range(1, len(right_str) + 1))
    for idx in idxs:
        if left_str.endswith(right_str[:idx]):
            return True
    return False
