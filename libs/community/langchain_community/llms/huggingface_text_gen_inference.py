import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_pydantic_field_names

logger = logging.getLogger(__name__)


@deprecated("0.0.21", removal="0.2.0", alternative="HuggingFaceEndpoint")
class HuggingFaceTextGenInference(LLM):
    """
    HuggingFace text generation API.
    ! This class is deprecated, you should use HuggingFaceEndpoint instead !

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
            print(llm("What is Deep Learning?"))  # noqa: T201

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
            print(llm("What is Deep Learning?"))  # noqa: T201

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

                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk

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

                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text)
                yield chunk

            # break if stop sequence found
            if stop_seq_found:
                break
