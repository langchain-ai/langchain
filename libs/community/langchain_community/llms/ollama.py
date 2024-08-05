from __future__ import annotations

import json
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra


def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return GenerationChunk(
        text=parsed_response.get("response", ""), generation_info=generation_info
    )


class OllamaEndpointNotFoundError(Exception):
    """Raised when the Ollama endpoint is not found."""


class _OllamaCommon(BaseLanguageModel):
    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""

    model: str = "llama2"
    """Model name to use."""

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict when generating text.
    (Default: 128, -1 = infinite generation, -2 = fill context)"""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    system: Optional[str] = None
    """system prompt (overrides what is defined in the Modelfile)"""

    template: Optional[str] = None
    """full prompt or prompt template (overrides what is defined in the Modelfile)"""

    format: Optional[str] = None
    """Specify the format of the output (e.g., json)"""

    timeout: Optional[int] = None
    """Timeout for the request stream"""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory.

    The parameter (Default: 5 minutes) can be set to:
    1. a duration string in Golang (such as "10m" or "24h");
    2. a number in seconds (such as 3600);
    3. any negative number which will keep the model loaded \
        in memory (e.g. -1 or "-1m");
    4. 0 which will unload the model immediately after generating a response;

    See the [Ollama documents](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)"""

    raw: Optional[bool] = None
    """raw or not."""

    headers: Optional[dict] = None
    """Additional headers to pass to endpoint (e.g. Authorization, Referer).
    This is useful when Ollama is hosted on cloud services that require
    tokens for authentication.
    """

    auth: Union[Callable, Tuple, None] = None
    """Additional auth tuple or callable to enable Basic/Digest/Custom HTTP Auth.
    Expects the same format, type and values as requests.request auth parameter."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "model": self.model,
            "format": self.format,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
            "system": self.system,
            "template": self.template,
            "keep_alive": self.keep_alive,
            "raw": self.raw,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model, "format": self.format}, **self._default_params}

    def _create_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        payload = {"prompt": prompt, "images": images}
        yield from self._create_stream(
            payload=payload,
            stop=stop,
            api_url=f"{self.base_url}/api/generate",
            **kwargs,
        )

    async def _acreate_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {"prompt": prompt, "images": images}
        async for item in self._acreate_stream(
            payload=payload,
            stop=stop,
            api_url=f"{self.base_url}/api/generate",
            **kwargs,
        ):
            yield item

    def _create_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                "stop": stop,
                **{k: v for k, v in kwargs.items() if k not in self._default_params},
            }

        if payload.get("messages"):
            request_payload = {"messages": payload.get("messages", []), **params}
        else:
            request_payload = {
                "prompt": payload.get("prompt"),
                "images": payload.get("images", []),
                **params,
            }
        response = requests.post(
            url=api_url,
            headers={
                "Content-Type": "application/json",
                **(self.headers if isinstance(self.headers, dict) else {}),
            },
            auth=self.auth,
            json=request_payload,
            stream=True,
            timeout=self.timeout,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            if response.status_code == 404:
                raise OllamaEndpointNotFoundError(
                    "Ollama call failed with status code 404. "
                    "Maybe your model is not found "
                    f"and you should pull the model with `ollama pull {self.model}`."
                )
            else:
                optional_detail = response.text
                raise ValueError(
                    f"Ollama call failed with status code {response.status_code}."
                    f" Details: {optional_detail}"
                )
        return response.iter_lines(decode_unicode=True)

    async def _acreate_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                "stop": stop,
                **{k: v for k, v in kwargs.items() if k not in self._default_params},
            }

        if payload.get("messages"):
            request_payload = {"messages": payload.get("messages", []), **params}
        else:
            request_payload = {
                "prompt": payload.get("prompt"),
                "images": payload.get("images", []),
                **params,
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=api_url,
                headers={
                    "Content-Type": "application/json",
                    **(self.headers if isinstance(self.headers, dict) else {}),
                },
                auth=self.auth,  # type: ignore[arg-type]
                json=request_payload,
                timeout=self.timeout,  # type: ignore[arg-type]
            ) as response:
                if response.status != 200:
                    if response.status == 404:
                        raise OllamaEndpointNotFoundError(
                            "Ollama call failed with status code 404."
                        )
                    else:
                        optional_detail = response.text
                        raise ValueError(
                            f"Ollama call failed with status code {response.status}."
                            f" Details: {optional_detail}"
                        )
                async for line in response.content:
                    yield line.decode("utf-8")

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    async def _astream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk


class Ollama(BaseLLM, _OllamaCommon):
    """Ollama locally runs large language models.
    To use, follow the instructions at https://ollama.ai/.
    Example:
        .. code-block:: python
            from langchain_community.llms import Ollama
            ollama = Ollama(model="llama2")
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ollama-llm"

    def _generate(  # type: ignore[override]
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Ollama's generate endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = ollama("Tell me a joke.")
        """
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            final_chunk = super()._stream_with_aggregation(
                prompt,
                stop=stop,
                images=images,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)  # type: ignore[arg-type]

    async def _agenerate(  # type: ignore[override]
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Ollama's generate endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = ollama("Tell me a joke.")
        """
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            final_chunk = await super()._astream_with_aggregation(
                prompt,
                stop=stop,
                images=images,
                run_manager=run_manager,  # type: ignore[arg-type]
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)  # type: ignore[arg-type]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_generation_chunk(stream_resp)
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk
