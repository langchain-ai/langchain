"""Ollama large language models."""

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
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


class OllamaLLM(BaseLLM):
    """OllamaLLM large language models.

    Example:
        .. code-block:: python

            from langchain_ollama import OllamaLLM

            model = OllamaLLM(model="llama3")
            model.invoke("Come up with 10 names for a song about parrots")
    """

    model: str
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

    format: Literal["", "json"] = ""
    """Specify the format of the output (options: json)"""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx Client. 
    For a full list of the params, see [this link](https://pydoc.dev/httpx/latest/httpx.Client.html)
    """

    _client: Client = PrivateAttr(default=None)  # type: ignore
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = PrivateAttr(default=None)  # type: ignore
    """
    The async client to use for making requests.
    """

    def _generate_params(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
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
                "stop": self.stop if stop is None else stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        )

        params = {
            "prompt": prompt,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "format": kwargs.pop("format", self.format),
            "options": Options(**options_dict),
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

        return params

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama-llm"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
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
        self._client = Client(host=self.base_url, **client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **client_kwargs)
        return self

    async def _acreate_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[Mapping[str, Any], str]]:
        async for part in await self._async_client.generate(
            **self._generate_params(prompt, stop=stop, **kwargs)
        ):  # type: ignore
            yield part  # type: ignore

    def _create_generate_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        yield from self._client.generate(
            **self._generate_params(prompt, stop=stop, **kwargs)
        )  # type: ignore

    async def _astream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk = None
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = GenerationChunk(
                    text=stream_resp["response"] if "response" in stream_resp else "",
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
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk = None
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = GenerationChunk(
                    text=stream_resp["response"] if "response" in stream_resp else "",
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
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = GenerationChunk(
                    text=(stream_resp.get("response", "")),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = GenerationChunk(
                    text=(stream_resp.get("response", "")),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk
