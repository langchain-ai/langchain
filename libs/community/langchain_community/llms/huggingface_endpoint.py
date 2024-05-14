import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names

logger = logging.getLogger(__name__)

VALID_TASKS = (
    "text2text-generation",
    "text-generation",
    "summarization",
    "conversational",
)


@deprecated(
    since="0.0.37",
    removal="0.3",
    alternative_import="from langchain_huggingface.llms import HuggingFaceEndpoint",
)
class HuggingFaceEndpoint(LLM):
    """
    HuggingFace Endpoint.

    To use this class, you should have installed the ``huggingface_hub`` package, and
    the environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token,
    or given as a named parameter to the constructor.

    Example:
        .. code-block:: python

            # Basic Example (no streaming)
            llm = HuggingFaceEndpoint(
                endpoint_url="http://localhost:8010/",
                max_new_tokens=512,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                huggingfacehub_api_token="my-api-key"
            )
            print(llm.invoke("What is Deep Learning?"))

            # Streaming response example
            from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

            callbacks = [StreamingStdOutCallbackHandler()]
            llm = HuggingFaceEndpoint(
                endpoint_url="http://localhost:8010/",
                max_new_tokens=512,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                callbacks=callbacks,
                streaming=True,
                huggingfacehub_api_token="my-api-key"
            )
            print(llm.invoke("What is Deep Learning?"))

    """  # noqa: E501

    endpoint_url: Optional[str] = None
    """Endpoint URL to use."""
    repo_id: Optional[str] = None
    """Repo to use."""
    huggingfacehub_api_token: Optional[str] = None
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
    model: str
    client: Any
    async_client: Any
    task: Optional[str] = None
    """Task to call the model with.
    Should be a task that returns `generated_text` or `summary_text`."""

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
                    Please make sure that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        if "endpoint_url" not in values and "repo_id" not in values:
            raise ValueError(
                "Please specify an `endpoint_url` or `repo_id` for the model."
            )
        if "endpoint_url" in values and "repo_id" in values:
            raise ValueError(
                "Please specify either an `endpoint_url` OR a `repo_id`, not both."
            )
        values["model"] = values.get("endpoint_url") or values.get("repo_id")
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that package is installed and that the API token is valid."""
        try:
            from huggingface_hub import login

        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        try:
            huggingfacehub_api_token = get_from_dict_or_env(
                values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
            )
            login(token=huggingfacehub_api_token)
        except Exception as e:
            raise ValueError(
                "Could not authenticate with huggingface_hub. "
                "Please check your API token."
            ) from e

        from huggingface_hub import AsyncInferenceClient, InferenceClient

        values["client"] = InferenceClient(
            model=values["model"],
            timeout=values["timeout"],
            token=huggingfacehub_api_token,
            **values["server_kwargs"],
        )
        values["async_client"] = AsyncInferenceClient(
            model=values["model"],
            timeout=values["timeout"],
            token=huggingfacehub_api_token,
            **values["server_kwargs"],
        )

        return values

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

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url, "task": self.task},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface_endpoint"

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
        """Call out to HuggingFace Hub's inference endpoint."""
        invocation_params = self._invocation_params(stop, **kwargs)
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **invocation_params):
                completion += chunk.text
            return completion
        else:
            invocation_params["stop"] = invocation_params[
                "stop_sequences"
            ]  # porting 'stop_sequences' into the 'stop' argument
            response = self.client.post(
                json={"inputs": prompt, "parameters": invocation_params},
                stream=False,
                task=self.task,
            )
            try:
                response_text = json.loads(response.decode())[0]["generated_text"]
            except KeyError:
                response_text = json.loads(response.decode())["generated_text"]

            # Maybe the generation has stopped at one of the stop sequences:
            # then we remove this stop sequence from the end of the generated text
            for stop_seq in invocation_params["stop_sequences"]:
                if response_text[-len(stop_seq) :] == stop_seq:
                    response_text = response_text[: -len(stop_seq)]
            return response_text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        invocation_params = self._invocation_params(stop, **kwargs)
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt, stop, run_manager, **invocation_params
            ):
                completion += chunk.text
            return completion
        else:
            invocation_params["stop"] = invocation_params["stop_sequences"]
            response = await self.async_client.post(
                json={"inputs": prompt, "parameters": invocation_params},
                stream=False,
                task=self.task,
            )
            try:
                response_text = json.loads(response.decode())[0]["generated_text"]
            except KeyError:
                response_text = json.loads(response.decode())["generated_text"]

            # Maybe the generation has stopped at one of the stop sequences:
            # then remove this stop sequence from the end of the generated text
            for stop_seq in invocation_params["stop_sequences"]:
                if response_text[-len(stop_seq) :] == stop_seq:
                    response_text = response_text[: -len(stop_seq)]
            return response_text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)

        for response in self.client.text_generation(
            prompt, **invocation_params, stream=True
        ):
            # identify stop sequence in generated text, if any
            stop_seq_found: Optional[str] = None
            for stop_seq in invocation_params["stop_sequences"]:
                if stop_seq in response:
                    stop_seq_found = stop_seq

            # identify text to yield
            text: Optional[str] = None
            if stop_seq_found:
                text = response[: response.index(stop_seq_found)]
            else:
                text = response

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
        async for response in await self.async_client.text_generation(
            prompt, **invocation_params, stream=True
        ):
            # identify stop sequence in generated text, if any
            stop_seq_found: Optional[str] = None
            for stop_seq in invocation_params["stop_sequences"]:
                if stop_seq in response:
                    stop_seq_found = stop_seq

            # identify text to yield
            text: Optional[str] = None
            if stop_seq_found:
                text = response[: response.index(stop_seq_found)]
            else:
                text = response

            # yield text, if any
            if text:
                chunk = GenerationChunk(text=text)

                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text)
                yield chunk

            # break if stop sequence found
            if stop_seq_found:
                break
