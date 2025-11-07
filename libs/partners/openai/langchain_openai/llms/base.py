"""Base classes for OpenAI large language models. Chat models are in `chat_models/`."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping
from typing import Any, Literal

import openai
import tiktoken
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


def _update_token_usage(
    keys: set[str], response: dict[str, Any], token_usage: dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def _stream_response_to_generation_chunk(
    stream_response: dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    if not stream_response["choices"]:
        return GenerationChunk(text="")
    return GenerationChunk(
        text=stream_response["choices"][0]["text"] or "",
        generation_info={
            "finish_reason": stream_response["choices"][0].get("finish_reason", None),
            "logprobs": stream_response["choices"][0].get("logprobs", None),
        },
    )


class BaseOpenAI(BaseLLM):
    """Base OpenAI large language model class.

    Setup:
        Install `langchain-openai` and set environment variable `OPENAI_API_KEY`.

        ```bash
        pip install -U langchain-openai
        export OPENAI_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model_name:
            Name of OpenAI model to use.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.
        top_p:
            Total probability mass of tokens to consider at each step.
        frequency_penalty:
            Penalizes repeated tokens according to frequency.
        presence_penalty:
            Penalizes repeated tokens.
        n:
            How many completions to generate for each prompt.
        best_of:
            Generates best_of completions server-side and returns the "best".
        logit_bias:
            Adjust the probability of specific tokens being generated.
        seed:
            Seed for generation.
        logprobs:
            Include the log probabilities on the logprobs most likely output tokens.
        streaming:
            Whether to stream the results or not.

    Key init args — client params:
        openai_api_key:
            OpenAI API key. If not passed in will be read from env var
            `OPENAI_API_KEY`.
        openai_api_base:
            Base URL path for API requests, leave blank if not using a proxy or
            service emulator.
        openai_organization:
            OpenAI organization ID. If not passed in will be read from env
            var `OPENAI_ORG_ID`.
        request_timeout:
            Timeout for requests to OpenAI completion API.
        max_retries:
            Maximum number of retries to make when generating.
        batch_size:
            Batch size to use when passing multiple documents to generate.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_openai.llms.base import BaseOpenAI

        model = BaseOpenAI(
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            # openai_api_key="...",
            # openai_api_base="...",
            # openai_organization="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        input_text = "The meaning of life is "
        response = model.invoke(input_text)
        print(response)
        ```

        ```txt
        "a philosophical question that has been debated by thinkers and
        scholars for centuries."
        ```

    Stream:
        ```python
        for chunk in model.stream(input_text):
            print(chunk, end="")
        ```
        ```txt
        a philosophical question that has been debated by thinkers and
        scholars for centuries.
        ```

    Async:
        ```python
        response = await model.ainvoke(input_text)

        # stream:
        # async for chunk in model.astream(input_text):
        #     print(chunk, end="")

        # batch:
        # await model.abatch([input_text])
        ```
        ```
        "a philosophical question that has been debated by thinkers and
        scholars for centuries."
        ```

    """

    client: Any = Field(default=None, exclude=True)

    async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(default="gpt-3.5-turbo-instruct", alias="model")
    """Model name to use."""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""

    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""

    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""

    presence_penalty: float = 0
    """Penalizes repeated tokens."""

    n: int = 1
    """How many completions to generate for each prompt."""

    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    openai_api_key: SecretStr | None | Callable[[], str] = Field(
        alias="api_key", default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""

    openai_api_base: str | None = Field(
        alias="base_url", default_factory=from_env("OPENAI_API_BASE", default=None)
    )
    """Base URL path for API requests, leave blank if not using a proxy or service
        emulator."""

    openai_organization: str | None = Field(
        alias="organization",
        default_factory=from_env(
            ["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None
        ),
    )
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""

    # to support explicit proxy for OpenAI
    openai_proxy: str | None = Field(
        default_factory=from_env("OPENAI_PROXY", default=None)
    )

    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, `httpx.Timeout` or
    None."""

    logit_bias: dict[str, float] | None = None
    """Adjust the probability of specific tokens being generated."""

    max_retries: int = 2
    """Maximum number of retries to make when generating."""

    seed: int | None = None
    """Seed for generation"""

    logprobs: int | None = None
    """Include the log probabilities on the logprobs most likely output tokens,
    as well the chosen tokens."""

    streaming: bool = False
    """Whether to stream the results or not."""

    allowed_special: Literal["all"] | set[str] = set()
    """Set of special tokens that are allowed。"""

    disallowed_special: Literal["all"] | Collection[str] = "all"
    """Set of special tokens that are not allowed。"""

    tiktoken_model_name: str | None = None
    """The model name to pass to tiktoken when using this class.
    Tiktoken is used to count the number of tokens in documents to constrain
    them to be under a certain limit. By default, when set to None, this will
    be the same as the embedding model name. However, there are some cases
    where you may want to use this Embedding class with a model name not
    supported by tiktoken. This can include when using Azure embeddings or
    when using one of the many model providers that expose an OpenAI-like
    API but with different models. In those cases, in order to avoid erroring
    when tiktoken is called, you can specify a model name to use here."""

    default_headers: Mapping[str, str] | None = None

    default_query: Mapping[str, object] | None = None

    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Any | None = None
    """Optional `httpx.Client`. Only used for sync invocations. Must specify
        `http_async_client` as well if you'd like a custom client for async
        invocations.
    """

    http_async_client: Any | None = None
    """Optional `httpx.AsyncClient`. Only used for async invocations. Must specify
        `http_client` as well if you'd like a custom client for sync invocations."""

    extra_body: Mapping[str, Any] | None = None
    """Optional additional JSON properties to include in the request parameters when
    making requests to OpenAI compatible APIs, such as vLLM."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.streaming and self.n > 1:
            msg = "Cannot stream results when n > 1."
            raise ValueError(msg)
        if self.streaming and self.best_of > 1:
            msg = "Cannot stream results when best_of > 1."
            raise ValueError(msg)

        # Resolve API key from SecretStr or Callable
        api_key_value: str | Callable[[], str] | None = None
        if self.openai_api_key is not None:
            if isinstance(self.openai_api_key, SecretStr):
                api_key_value = self.openai_api_key.get_secret_value()
            elif callable(self.openai_api_key):
                api_key_value = self.openai_api_key

        client_params: dict = {
            "api_key": api_key_value,
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not self.client:
            sync_specific = {"http_client": self.http_client}
            self.client = openai.OpenAI(**client_params, **sync_specific).completions  # type: ignore[arg-type]
        if not self.async_client:
            async_specific = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            ).completions

        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "seed": self.seed,
            "logprobs": self.logprobs,
        }

        if self.logit_bias is not None:
            normal_params["logit_bias"] = self.logit_bias

        if self.max_tokens is not None:
            normal_params["max_tokens"] = self.max_tokens

        if self.extra_body is not None:
            normal_params["extra_body"] = self.extra_body

        # Azure gpt-35-turbo doesn't support best_of
        # don't specify best_of if it is 1
        if self.best_of > 1:
            normal_params["best_of"] = self.best_of

        return {**normal_params, **self.model_kwargs}

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**self._invocation_params, **kwargs, "stream": True}
        self.get_sub_prompts(params, [prompt], stop)  # this mutates params
        for stream_resp in self.client.create(prompt=prompt, **params):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.model_dump()
            chunk = _stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                    logprobs=(
                        chunk.generation_info["logprobs"]
                        if chunk.generation_info
                        else None
                    ),
                )
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = {**self._invocation_params, **kwargs, "stream": True}
        self.get_sub_prompts(params, [prompt], stop)  # this mutates params
        async for stream_resp in await self.async_client.create(
            prompt=prompt, **params
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.model_dump()
            chunk = _stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                    logprobs=(
                        chunk.generation_info["logprobs"]
                        if chunk.generation_info
                        else None
                    ),
                )
            yield chunk

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use for the call.

        Returns:
            The full LLM output.

        Example:
            ```python
            response = openai.generate(["Tell me a joke."])
            ```
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: str | None = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    msg = "Cannot stream results with multiple prompts."
                    raise ValueError(msg)

                generation: GenerationChunk | None = None
                for chunk in self._stream(_prompts[0], stop, run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                if generation is None:
                    msg = "Generation is empty after streaming."
                    raise ValueError(msg)
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason")
                            if generation.generation_info
                            else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs")
                            if generation.generation_info
                            else None
                        ),
                    }
                )
            else:
                response = self.client.create(prompt=_prompts, **params)
                if not isinstance(response, dict):
                    # V1 client returns the response in an PyDantic object instead of
                    # dict. For the transition period, we deep convert it to dict.
                    response = response.model_dump()

                # Sometimes the AI Model calling will get error, we should raise it.
                # Otherwise, the next code 'choices.extend(response["choices"])'
                # will throw a "TypeError: 'NoneType' object is not iterable" error
                # to mask the true error. Because 'response["choices"]' is None.
                if response.get("error"):
                    raise ValueError(response.get("error"))

                choices.extend(response["choices"])
                _update_token_usage(_keys, response, token_usage)
                if not system_fingerprint:
                    system_fingerprint = response.get("system_fingerprint")
        return self.create_llm_result(
            choices, prompts, params, token_usage, system_fingerprint=system_fingerprint
        )

    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to OpenAI's endpoint async with k unique prompts."""
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: str | None = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    msg = "Cannot stream results with multiple prompts."
                    raise ValueError(msg)

                generation: GenerationChunk | None = None
                async for chunk in self._astream(
                    _prompts[0], stop, run_manager, **kwargs
                ):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                if generation is None:
                    msg = "Generation is empty after streaming."
                    raise ValueError(msg)
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason")
                            if generation.generation_info
                            else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs")
                            if generation.generation_info
                            else None
                        ),
                    }
                )
            else:
                response = await self.async_client.create(prompt=_prompts, **params)
                if not isinstance(response, dict):
                    response = response.model_dump()
                choices.extend(response["choices"])
                _update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(
            choices, prompts, params, token_usage, system_fingerprint=system_fingerprint
        )

    def get_sub_prompts(
        self,
        params: dict[str, Any],
        prompts: list[str],
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Get the sub prompts for llm call."""
        if stop is not None:
            params["stop"] = stop
        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                msg = "max_tokens set to -1 not supported for multiple inputs."
                raise ValueError(msg)
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        return [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]

    def create_llm_result(
        self,
        choices: Any,
        prompts: list[str],
        params: dict[str, Any],
        token_usage: dict[str, int],
        *,
        system_fingerprint: str | None = None,
    ) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        n = params.get("n", self.n)
        for i, _ in enumerate(prompts):
            sub_choices = choices[i * n : (i + 1) * n]
            generations.append(
                [
                    Generation(
                        text=choice["text"],
                        generation_info={
                            "finish_reason": choice.get("finish_reason"),
                            "logprobs": choice.get("logprobs"),
                        },
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        if system_fingerprint:
            llm_output["system_fingerprint"] = system_fingerprint
        return LLMResult(generations=generations, llm_output=llm_output)

    @property
    def _invocation_params(self) -> dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai"

    def get_token_ids(self, text: str) -> list[int]:
        """Get the token IDs using the tiktoken package."""
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        # tiktoken NOT supported for Python < 3.8
        if sys.version_info[1] < 8:
            return super().get_num_tokens(text)

        model_name = self.tiktoken_model_name or self.model_name
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        return enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    @staticmethod
    def modelname_to_contextsize(modelname: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a model.

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size

        Example:
            ```python
            max_tokens = openai.modelname_to_contextsize("gpt-3.5-turbo-instruct")
            ```
        """
        model_token_mapping = {
            "gpt-4o-mini": 128_000,
            "gpt-4o": 128_000,
            "gpt-4o-2024-05-13": 128_000,
            "gpt-4": 8192,
            "gpt-4-0314": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0314": 32768,
            "gpt-4-32k-0613": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0301": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-16k-0613": 16385,
            "gpt-3.5-turbo-instruct": 4096,
            "text-ada-001": 2049,
            "ada": 2049,
            "text-babbage-001": 2040,
            "babbage": 2049,
            "text-curie-001": 2049,
            "curie": 2049,
            "davinci": 2049,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "code-davinci-002": 8001,
            "code-davinci-001": 8001,
            "code-cushman-002": 2048,
            "code-cushman-001": 2048,
        }

        # handling finetuned models
        if "ft-" in modelname:
            modelname = modelname.split(":")[0]

        context_size = model_token_mapping.get(modelname)

        if context_size is None:
            raise ValueError(
                f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
                "Known models are: " + ", ".join(model_token_mapping.keys())
            )

        return context_size

    @property
    def max_context_size(self) -> int:
        """Get max context size for this model."""
        return self.modelname_to_contextsize(self.model_name)

    def max_tokens_for_prompt(self, prompt: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a prompt.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The maximum number of tokens to generate for a prompt.

        Example:
            ```python
            max_tokens = openai.max_tokens_for_prompt("Tell me a joke.")
            ```
        """
        num_tokens = self.get_num_tokens(prompt)
        return self.max_context_size - num_tokens


class OpenAI(BaseOpenAI):
    """OpenAI completion model integration.

    Setup:
        Install `langchain-openai` and set environment variable `OPENAI_API_KEY`.

        ```bash
        pip install -U langchain-openai
        export OPENAI_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of OpenAI model to use.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.
        logprobs:
            Whether to return logprobs.
        stream_options:
            Configure streaming outputs, like whether to return token usage when
            streaming (`{"include_usage": True}`).

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            OpenAI API key. If not passed in will be read from env var `OPENAI_API_KEY`.
        base_url:
            Base URL for API requests. Only specify if using a proxy or service
            emulator.
        organization:
            OpenAI organization ID. If not passed in will be read from env
            var `OPENAI_ORG_ID`.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_openai import OpenAI

        model = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0,
            max_retries=2,
            # api_key="...",
            # base_url="...",
            # organization="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        input_text = "The meaning of life is "
        model.invoke(input_text)
        ```
        ```txt
        "a philosophical question that has been debated by thinkers and scholars for centuries."
        ```

    Stream:
        ```python
        for chunk in model.stream(input_text):
            print(chunk, end="|")
        ```
        ```txt
        a| philosophical| question| that| has| been| debated| by| thinkers| and| scholars| for| centuries|.
        ```

        ```python
        "".join(model.stream(input_text))
        ```
        ```txt
        "a philosophical question that has been debated by thinkers and scholars for centuries."
        ```

    Async:
        ```python
        await model.ainvoke(input_text)

        # stream:
        # async for chunk in (await model.astream(input_text)):
        #    print(chunk)

        # batch:
        # await model.abatch([input_text])
        ```
        ```txt
        "a philosophical question that has been debated by thinkers and scholars for centuries."
        ```
    """  # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "llms", "openai"]`
        """
        return ["langchain", "llms", "openai"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _invocation_params(self) -> dict[str, Any]:
        return {"model": self.model_name, **super()._invocation_params}

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret keys to environment variables."""
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """LangChain attributes for this class."""
        attributes: dict[str, Any] = {}
        if self.openai_api_base:
            attributes["openai_api_base"] = self.openai_api_base

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.openai_proxy:
            attributes["openai_proxy"] = self.openai_proxy

        return attributes
