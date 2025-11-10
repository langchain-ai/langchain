"""Factory functions for chat models."""

from __future__ import annotations

import warnings
from importlib import util
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from langchain_core.runnables.schema import StreamEvent
    from langchain_core.tools import BaseTool
    from langchain_core.tracers import RunLog, RunLogPatch
    from pydantic import BaseModel


@overload
def init_chat_model(
    model: str,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel: ...


@overload
def init_chat_model(
    model: None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...


@overload
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = ...,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...


# FOR CONTRIBUTORS: If adding support for a new provider, please append the provider
# name to the supported list in the docstring below. Do *not* change the order of the
# existing providers.
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel:
    """Initialize a chat model from any supported provider using a unified interface.

    **Two main use cases:**

    1. **Fixed model** – specify the model upfront and get a ready-to-use chat model.
    2. **Configurable model** – choose to specify parameters (including model name) at
        runtime via `config`. Makes it easy to switch between models/providers without
        changing your code

    !!! note
        Requires the integration package for the chosen model provider to be installed.

        See the `model_provider` parameter below for specific package names
        (e.g., `pip install langchain-openai`).

        Refer to the [provider integration's API reference](https://docs.langchain.com/oss/python/integrations/providers)
        for supported model parameters to use as `**kwargs`.

    Args:
        model: The name or ID of the model, e.g. `'o3-mini'`, `'claude-sonnet-4-5-20250929'`.

            You can also specify model and model provider in a single argument using
            `'{model_provider}:{model}'` format, e.g. `'openai:o1'`.

            Will attempt to infer `model_provider` from model if not specified.

            The following providers will be inferred based on these model prefixes:

            - `gpt-...` | `o1...` | `o3...`       -> `openai`
            - `claude...`                         -> `anthropic`
            - `amazon...`                         -> `bedrock`
            - `gemini...`                         -> `google_vertexai`
            - `command...`                        -> `cohere`
            - `accounts/fireworks...`             -> `fireworks`
            - `mistral...`                        -> `mistralai`
            - `deepseek...`                       -> `deepseek`
            - `grok...`                           -> `xai`
            - `sonar...`                          -> `perplexity`
        model_provider: The model provider if not specified as part of the model arg
            (see above).

            Supported `model_provider` values and the corresponding integration package
            are:

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `anthropic`               -> [`langchain-anthropic`](https://docs.langchain.com/oss/python/integrations/providers/anthropic)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_ai`                -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `google_genai`            -> [`langchain-google-genai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `bedrock_converse`        -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `fireworks`               -> [`langchain-fireworks`](https://docs.langchain.com/oss/python/integrations/providers/fireworks)
            - `together`                -> [`langchain-together`](https://docs.langchain.com/oss/python/integrations/providers/together)
            - `mistralai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `groq`                    -> [`langchain-groq`](https://docs.langchain.com/oss/python/integrations/providers/groq)
            - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)
            - `google_anthropic_vertex` -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `deepseek`                -> [`langchain-deepseek`](https://docs.langchain.com/oss/python/integrations/providers/deepseek)
            - `ibm`                     -> [`langchain-ibm`](https://docs.langchain.com/oss/python/integrations/providers/deepseek)
            - `nvidia`                  -> [`langchain-nvidia-ai-endpoints`](https://docs.langchain.com/oss/python/integrations/providers/nvidia)
            - `xai`                     -> [`langchain-xai`](https://docs.langchain.com/oss/python/integrations/providers/xai)
            - `perplexity`              -> [`langchain-perplexity`](https://docs.langchain.com/oss/python/integrations/providers/perplexity)

        configurable_fields: Which model parameters are configurable at runtime:

            - `None`: No configurable fields (i.e., a fixed model).
            - `'any'`: All fields are configurable. **See security note below.**
            - `list[str] | Tuple[str, ...]`: Specified fields are configurable.

            Fields are assumed to have `config_prefix` stripped if a `config_prefix` is
            specified.

            If `model` is specified, then defaults to `None`.

            If `model` is not specified, then defaults to `("model", "model_provider")`.

            !!! warning "Security note"

                Setting `configurable_fields="any"` means fields like `api_key`,
                `base_url`, etc., can be altered at runtime, potentially redirecting
                model requests to a different service/user.

                Make sure that if you're accepting untrusted configurations that you
                enumerate the `configurable_fields=(...)` explicitly.

        config_prefix: Optional prefix for configuration keys.

            Useful when you have multiple configurable models in the same application.

            If `'config_prefix'` is a non-empty string then `model` will be configurable
            at runtime via the `config["configurable"]["{config_prefix}_{param}"]` keys.
            See examples below.

            If `'config_prefix'` is an empty string then model will be configurable via
            `config["configurable"]["{param}"]`.
        **kwargs: Additional model-specific keyword args to pass to the underlying
            chat model's `__init__` method. Common parameters include:

            - `temperature`: Model temperature for controlling randomness.
            - `max_tokens`: Maximum number of output tokens.
            - `timeout`: Maximum time (in seconds) to wait for a response.
            - `max_retries`: Maximum number of retry attempts for failed requests.
            - `base_url`: Custom API endpoint URL.
            - `rate_limiter`: A
                [`BaseRateLimiter`][langchain_core.rate_limiters.BaseRateLimiter]
                instance to control request rate.

            Refer to the specific model provider's
            [integration reference](https://reference.langchain.com/python/integrations/)
            for all available parameters.

    Returns:
        A `BaseChatModel` corresponding to the `model_name` and `model_provider`
            specified if configurability is inferred to be `False`. If configurable, a
            chat model emulator that initializes the underlying model at runtime once a
            config is passed in.

    Raises:
        ValueError: If `model_provider` cannot be inferred or isn't supported.
        ImportError: If the model provider integration package is not installed.

    ???+ example "Initialize a non-configurable model"

        ```python
        # pip install langchain langchain-openai langchain-anthropic langchain-google-vertexai

        from langchain.chat_models import init_chat_model

        o3_mini = init_chat_model("openai:o3-mini", temperature=0)
        claude_sonnet = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=0)
        gemini_2-5_flash = init_chat_model("google_vertexai:gemini-2.5-flash", temperature=0)

        o3_mini.invoke("what's your name")
        claude_sonnet.invoke("what's your name")
        gemini_2-5_flash.invoke("what's your name")
        ```

    ??? example "Partially configurable model with no default"

        ```python
        # pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model

        # (We don't need to specify configurable=True if a model isn't specified.)
        configurable_model = init_chat_model(temperature=0)

        configurable_model.invoke("what's your name", config={"configurable": {"model": "gpt-4o"}})
        # Use GPT-4o to generate the response

        configurable_model.invoke(
            "what's your name",
            config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},
        )
        ```

    ??? example "Fully configurable model with a default"

        ```python
        # pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model

        configurable_model_with_default = init_chat_model(
            "openai:gpt-4o",
            configurable_fields="any",  # This allows us to configure other params like temperature, max_tokens, etc at runtime.
            config_prefix="foo",
            temperature=0,
        )

        configurable_model_with_default.invoke("what's your name")
        # GPT-4o response with temperature 0 (as set in default)

        configurable_model_with_default.invoke(
            "what's your name",
            config={
                "configurable": {
                    "foo_model": "anthropic:claude-sonnet-4-5-20250929",
                    "foo_temperature": 0.6,
                }
            },
        )
        # Override default to use Sonnet 4.5 with temperature 0.6 to generate response
        ```

    ??? example "Bind tools to a configurable model"

        You can call any chat model declarative methods on a configurable model in the
        same way that you would with a normal model:

        ```python
        # pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        configurable_model = init_chat_model(
            "gpt-4o", configurable_fields=("model", "model_provider"), temperature=0
        )

        configurable_model_with_tools = configurable_model.bind_tools(
            [
                GetWeather,
                GetPopulation,
            ]
        )
        configurable_model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        # Use GPT-4o

        configurable_model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?",
            config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},
        )
        # Use Sonnet 4.5
        ```

    """  # noqa: E501
    if not model and not configurable_fields:
        configurable_fields = ("model", "model_provider")
    config_prefix = config_prefix or ""
    if config_prefix and not configurable_fields:
        warnings.warn(
            f"{config_prefix=} has been set but no fields are configurable. Set "
            f"`configurable_fields=(...)` to specify the model params that are "
            f"configurable.",
            stacklevel=2,
        )

    if not configurable_fields:
        return _init_chat_model_helper(
            cast("str", model),
            model_provider=model_provider,
            **kwargs,
        )
    if model:
        kwargs["model"] = model
    if model_provider:
        kwargs["model_provider"] = model_provider
    return _ConfigurableModel(
        default_config=kwargs,
        config_prefix=config_prefix,
        configurable_fields=configurable_fields,
    )


def _init_chat_model_helper(
    model: str,
    *,
    model_provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    model, model_provider = _parse_model(model, model_provider)
    if model_provider == "openai":
        _check_pkg("langchain_openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, **kwargs)
    if model_provider == "anthropic":
        _check_pkg("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, **kwargs)  # type: ignore[call-arg,unused-ignore]
    if model_provider == "azure_openai":
        _check_pkg("langchain_openai")
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(model=model, **kwargs)
    if model_provider == "azure_ai":
        _check_pkg("langchain_azure_ai")
        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

        return AzureAIChatCompletionsModel(model=model, **kwargs)
    if model_provider == "cohere":
        _check_pkg("langchain_cohere")
        from langchain_cohere import ChatCohere

        return ChatCohere(model=model, **kwargs)
    if model_provider == "google_vertexai":
        _check_pkg("langchain_google_vertexai")
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(model=model, **kwargs)
    if model_provider == "google_genai":
        _check_pkg("langchain_google_genai")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, **kwargs)
    if model_provider == "fireworks":
        _check_pkg("langchain_fireworks")
        from langchain_fireworks import ChatFireworks

        return ChatFireworks(model=model, **kwargs)
    if model_provider == "ollama":
        try:
            _check_pkg("langchain_ollama")
            from langchain_ollama import ChatOllama
        except ImportError:
            # For backwards compatibility
            try:
                _check_pkg("langchain_community")
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                # If both langchain-ollama and langchain-community aren't available,
                # raise an error related to langchain-ollama
                _check_pkg("langchain_ollama")

        return ChatOllama(model=model, **kwargs)
    if model_provider == "together":
        _check_pkg("langchain_together")
        from langchain_together import ChatTogether

        return ChatTogether(model=model, **kwargs)
    if model_provider == "mistralai":
        _check_pkg("langchain_mistralai")
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=model, **kwargs)  # type: ignore[call-arg,unused-ignore]
    if model_provider == "huggingface":
        _check_pkg("langchain_huggingface")
        from langchain_huggingface import ChatHuggingFace

        return ChatHuggingFace(model_id=model, **kwargs)
    if model_provider == "groq":
        _check_pkg("langchain_groq")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, **kwargs)
    if model_provider == "bedrock":
        _check_pkg("langchain_aws")
        from langchain_aws import ChatBedrock

        return ChatBedrock(model_id=model, **kwargs)
    if model_provider == "bedrock_converse":
        _check_pkg("langchain_aws")
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(model=model, **kwargs)
    if model_provider == "google_anthropic_vertex":
        _check_pkg("langchain_google_vertexai")
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        return ChatAnthropicVertex(model=model, **kwargs)
    if model_provider == "deepseek":
        _check_pkg("langchain_deepseek", pkg_kebab="langchain-deepseek")
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(model=model, **kwargs)
    if model_provider == "nvidia":
        _check_pkg("langchain_nvidia_ai_endpoints")
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(model=model, **kwargs)
    if model_provider == "ibm":
        _check_pkg("langchain_ibm")
        from langchain_ibm import ChatWatsonx

        return ChatWatsonx(model_id=model, **kwargs)
    if model_provider == "xai":
        _check_pkg("langchain_xai")
        from langchain_xai import ChatXAI

        return ChatXAI(model=model, **kwargs)
    if model_provider == "perplexity":
        _check_pkg("langchain_perplexity")
        from langchain_perplexity import ChatPerplexity

        return ChatPerplexity(model=model, **kwargs)
    supported = ", ".join(_SUPPORTED_PROVIDERS)
    msg = f"Unsupported {model_provider=}.\n\nSupported model providers are: {supported}"
    raise ValueError(msg)


_SUPPORTED_PROVIDERS = {
    "openai",
    "anthropic",
    "azure_openai",
    "azure_ai",
    "cohere",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "bedrock",
    "bedrock_converse",
    "google_anthropic_vertex",
    "deepseek",
    "ibm",
    "xai",
    "perplexity",
}


def _attempt_infer_model_provider(model_name: str) -> str | None:
    if any(model_name.startswith(pre) for pre in ("gpt-", "o1", "o3")):
        return "openai"
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("command"):
        return "cohere"
    if model_name.startswith("accounts/fireworks"):
        return "fireworks"
    if model_name.startswith("gemini"):
        return "google_vertexai"
    if model_name.startswith("amazon."):
        return "bedrock"
    if model_name.startswith("mistral"):
        return "mistralai"
    if model_name.startswith("deepseek"):
        return "deepseek"
    if model_name.startswith("grok"):
        return "xai"
    if model_name.startswith("sonar"):
        return "perplexity"
    return None


def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    if not model_provider and ":" in model and model.split(":")[0] in _SUPPORTED_PROVIDERS:
        model_provider = model.split(":")[0]
        model = ":".join(model.split(":")[1:])
    model_provider = model_provider or _attempt_infer_model_provider(model)
    if not model_provider:
        msg = (
            f"Unable to infer model provider for {model=}, please specify model_provider directly."
        )
        raise ValueError(msg)
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider


def _check_pkg(pkg: str, *, pkg_kebab: str | None = None) -> None:
    if not util.find_spec(pkg):
        pkg_kebab = pkg_kebab if pkg_kebab is not None else pkg.replace("_", "-")
        msg = f"Unable to import {pkg}. Please install with `pip install -U {pkg_kebab}`"
        raise ImportError(msg)


def _remove_prefix(s: str, prefix: str) -> str:
    return s.removeprefix(prefix)


_DECLARATIVE_METHODS = ("bind_tools", "with_structured_output")


class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    def __init__(
        self,
        *,
        default_config: dict | None = None,
        configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[tuple[str, tuple, dict]] = (),
    ) -> None:
        self._default_config: dict = default_config or {}
        self._configurable_fields: Literal["any"] | list[str] = (
            configurable_fields if configurable_fields == "any" else list(configurable_fields)
        )
        self._config_prefix = (
            config_prefix + "_"
            if config_prefix and not config_prefix.endswith("_")
            else config_prefix
        )
        self._queued_declarative_operations: list[tuple[str, tuple, dict]] = list(
            queued_declarative_operations,
        )

    def __getattr__(self, name: str) -> Any:
        if name in _DECLARATIVE_METHODS:
            # Declarative operations that cannot be applied until after an actual model
            # object is instantiated. So instead of returning the actual operation,
            # we record the operation and its arguments in a queue. This queue is
            # then applied in order whenever we actually instantiate the model (in
            # self._model()).
            def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
                queued_declarative_operations = list(
                    self._queued_declarative_operations,
                )
                queued_declarative_operations.append((name, args, kwargs))
                return _ConfigurableModel(
                    default_config=dict(self._default_config),
                    configurable_fields=list(self._configurable_fields)
                    if isinstance(self._configurable_fields, list)
                    else self._configurable_fields,
                    config_prefix=self._config_prefix,
                    queued_declarative_operations=queued_declarative_operations,
                )

            return queue
        if self._default_config and (model := self._model()) and hasattr(model, name):
            return getattr(model, name)
        msg = f"{name} is not a BaseChatModel attribute"
        if self._default_config:
            msg += " and is not implemented on the default model"
        msg += "."
        raise AttributeError(msg)

    def _model(self, config: RunnableConfig | None = None) -> Runnable:
        params = {**self._default_config, **self._model_params(config)}
        model = _init_chat_model_helper(**params)
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model

    def _model_params(self, config: RunnableConfig | None) -> dict:
        config = ensure_config(config)
        model_params = {
            _remove_prefix(k, self._config_prefix): v
            for k, v in config.get("configurable", {}).items()
            if k.startswith(self._config_prefix)
        }
        if self._configurable_fields != "any":
            model_params = {k: v for k, v in model_params.items() if k in self._configurable_fields}
        return model_params

    def with_config(
        self,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> _ConfigurableModel:
        """Bind config to a `Runnable`, returning a new `Runnable`."""
        config = RunnableConfig(**(config or {}), **cast("RunnableConfig", kwargs))
        model_params = self._model_params(config)
        remaining_config = {k: v for k, v in config.items() if k != "configurable"}
        remaining_config["configurable"] = {
            k: v
            for k, v in config.get("configurable", {}).items()
            if _remove_prefix(k, self._config_prefix) not in model_params
        }
        queued_declarative_operations = list(self._queued_declarative_operations)
        if remaining_config:
            queued_declarative_operations.append(
                (
                    "with_config",
                    (),
                    {"config": remaining_config},
                ),
            )
        return _ConfigurableModel(
            default_config={**self._default_config, **model_params},
            configurable_fields=list(self._configurable_fields)
            if isinstance(self._configurable_fields, list)
            else self._configurable_fields,
            config_prefix=self._config_prefix,
            queued_declarative_operations=queued_declarative_operations,
        )

    @property
    def InputType(self) -> TypeAlias:
        """Get the input type for this `Runnable`."""
        from langchain_core.prompt_values import ChatPromptValueConcrete, StringPromptValue

        # This is a version of LanguageModelInput which replaces the abstract
        # base class BaseMessage with a union of its subclasses, which makes
        # for a much better schema.
        return str | StringPromptValue | ChatPromptValueConcrete | list[AnyMessage]

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._model(config).invoke(input, config=config, **kwargs)

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._model(config).ainvoke(input, config=config, **kwargs)

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        yield from self._model(config).stream(input, config=config, **kwargs)

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        async for x in self._model(config).astream(input, config=config, **kwargs):
            yield x

    def batch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Any]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return self._model(config).batch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # in parallel.
        return super().batch(
            inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )

    async def abatch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Any]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return await self._model(config).abatch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # in parallel.
        return await super().abatch(
            inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )

    def batch_as_completed(
        self,
        inputs: Sequence[LanguageModelInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Any | Exception]]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            yield from self._model(cast("RunnableConfig", config)).batch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # in parallel.
        else:
            yield from super().batch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )

    async def abatch_as_completed(
        self,
        inputs: Sequence[LanguageModelInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, Any]]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            async for x in self._model(
                cast("RunnableConfig", config),
            ).abatch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            ):
                yield x
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # in parallel.
        else:
            async for x in super().abatch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            ):
                yield x

    @override
    def transform(
        self,
        input: Iterator[LanguageModelInput],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        yield from self._model(config).transform(input, config=config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[LanguageModelInput],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        async for x in self._model(config).atransform(input, config=config, **kwargs):
            yield x

    @overload
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[True] = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch]: ...

    @overload
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[False],
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLog]: ...

    @override
    async def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: bool = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
        async for x in self._model(config).astream_log(  # type: ignore[call-overload, misc]
            input,
            config=config,
            diff=diff,
            with_streamed_output_list=with_streamed_output_list,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            **kwargs,
        ):
            yield x

    @override
    async def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        async for x in self._model(config).astream_events(
            input,
            config=config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            **kwargs,
        ):
            yield x

    # Explicitly added to satisfy downstream linters.
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.__getattr__("bind_tools")(tools, **kwargs)

    # Explicitly added to satisfy downstream linters.
    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        return self.__getattr__("with_structured_output")(schema, **kwargs)
