"""Ollama large language models."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import (
    Any,
    Literal,
    Optional,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM, LangSmithParams
from langchain_core.outputs import GenerationChunk, LLMResult
from ollama import AsyncClient, Client, Options
from pydantic import PrivateAttr, model_validator
from typing_extensions import Self

from ._utils import validate_model


class OllamaLLM(BaseLLM):
    """OllamaLLM large language models.

    Example:
        .. code-block:: python

            from langchain_ollama import OllamaLLM

            model = OllamaLLM(model="llama3")
            print(model.invoke("Come up with 10 names for a song about parrots"))
    """

    model: str
    """Model name to use."""

    reasoning: Optional[bool] = None
    """Controls the reasoning/thinking mode for
    `supported models <https://ollama.com/search?c=thinking>`__.

    - ``True``: Enables reasoning mode. The model's reasoning process will be
      captured and returned separately in the ``additional_kwargs`` of the
      response message, under ``reasoning_content``. The main response
      content will not include the reasoning tags.
    - ``False``: Disables reasoning mode. The model will not perform any reasoning,
      and the response will not include any reasoning content.
    - ``None`` (Default): The model will use its default reasoning behavior. If
      the model performs reasoning, the ``<think>`` and ``</think>`` tags will
      be present directly within the main response content."""

    validate_model_on_init: bool = False
    """Whether to validate the model exists in ollama locally on initialization.

    .. versionadded:: 0.3.4
    """

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
    next token. (Default: 2048)"""

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

    seed: Optional[int] = None
    """Sets the random number seed to use for generation. Setting this
    to a specific number will make the model generate the same text for
    the same prompt."""

    stop: Optional[list[str]] = None
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

    format: Literal["", "json"] = ""
    """Specify the format of the output (options: json)"""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx clients.
    These arguments are passed to both synchronous and async clients.
    Use sync_client_kwargs and async_client_kwargs to pass different arguments
    to synchronous and asynchronous clients.
    """

    async_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before passing to the HTTPX
    AsyncClient.

    For a full list of the params, see the `HTTPX documentation <https://www.python-httpx.org/api/#asyncclient>`__.
    """

    sync_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before passing to the HTTPX Client.

    For a full list of the params, see the `HTTPX documentation <https://www.python-httpx.org/api/#client>`__.
    """

    _client: Optional[Client] = PrivateAttr(default=None)
    """
    The client to use for making requests.
    """

    _async_client: Optional[AsyncClient] = PrivateAttr(default=None)
    """
    The async client to use for making requests.
    """

    def _generate_params(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.stop is not None and stop is not None:
            msg = "`stop` found in both the input and default params."
            raise ValueError(msg)
        if self.stop is not None:
            stop = self.stop

        options_dict = kwargs.pop(
            "options",
            {
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
                "seed": self.seed,
                "stop": self.stop if stop is None else stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        )

        return {
            "prompt": prompt,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "think": kwargs.pop("reasoning", self.reasoning),
            "format": kwargs.pop("format", self.format),
            "options": Options(**options_dict),
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama-llm"

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        if max_tokens := kwargs.get("num_predict", self.num_predict):
            params["ls_max_tokens"] = max_tokens
        return params

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients to use for ollama."""
        client_kwargs = self.client_kwargs or {}

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=self.base_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **async_client_kwargs)
        if self.validate_model_on_init:
            validate_model(self._client, self.model)
        return self

    async def _acreate_generate_stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[Mapping[str, Any], str]]:
        if self._async_client:
            async for part in await self._async_client.generate(
                **self._generate_params(prompt, stop=stop, **kwargs)
            ):
                yield part

    def _create_generate_stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        if self._client:
            yield from self._client.generate(
                **self._generate_params(prompt, stop=stop, **kwargs)
            )

    async def _astream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk = None
        thinking_content = ""
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                if stream_resp.get("thinking"):
                    thinking_content += stream_resp["thinking"]
                chunk = GenerationChunk(
                    text=stream_resp.get("response", ""),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        if thinking_content:
            if final_chunk.generation_info:
                final_chunk.generation_info["thinking"] = thinking_content
            else:
                final_chunk.generation_info = {"thinking": thinking_content}

        return final_chunk

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk = None
        thinking_content = ""
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                if stream_resp.get("thinking"):
                    thinking_content += stream_resp["thinking"]
                chunk = GenerationChunk(
                    text=stream_resp.get("response", ""),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        if thinking_content:
            if final_chunk.generation_info:
                final_chunk.generation_info["thinking"] = thinking_content
            else:
                final_chunk.generation_info = {"thinking": thinking_content}

        return final_chunk

    def _generate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            final_chunk = self._stream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)  # type: ignore[arg-type]

    async def _agenerate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            final_chunk = await self._astream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)  # type: ignore[arg-type]

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                additional_kwargs = {}
                if reasoning and (thinking_content := stream_resp.get("thinking")):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = GenerationChunk(
                    text=(stream_resp.get("response", "")),
                    generation_info={
                        "finish_reason": self.stop,
                        **additional_kwargs,
                        **(
                            dict(stream_resp) if stream_resp.get("done") is True else {}
                        ),
                    },
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                additional_kwargs = {}
                if reasoning and (thinking_content := stream_resp.get("thinking")):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = GenerationChunk(
                    text=(stream_resp.get("response", "")),
                    generation_info={
                        "finish_reason": self.stop,
                        **additional_kwargs,
                        **(
                            dict(stream_resp) if stream_resp.get("done") is True else {}
                        ),
                    },
                )
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk
