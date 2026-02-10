"""Factory functions for chat models."""

from __future__ import annotations

import functools
import importlib
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    cast,
    overload,
)

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompt_values import ChatPromptValueConcrete, StringPromptValue
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence
    from types import ModuleType

    from langchain_core.runnables.schema import StreamEvent
    from langchain_core.tools import BaseTool
    from langchain_core.tracers import RunLog, RunLogPatch
    from pydantic import BaseModel


def _call(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    # TODO: replace with operator.call when lower bounding to Python 3.11
    return cls(**kwargs)


_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    "anthropic": ("langchain_anthropic", "ChatAnthropic", _call),
    "azure_ai": ("langchain_azure_ai.chat_models", "AzureAIChatCompletionsModel", _call),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI", _call),
    "bedrock": ("langchain_aws", "ChatBedrock", _call),
    "bedrock_converse": ("langchain_aws", "ChatBedrockConverse", _call),
    "cohere": ("langchain_cohere", "ChatCohere", _call),
    "deepseek": ("langchain_deepseek", "ChatDeepSeek", _call),
    "fireworks": ("langchain_fireworks", "ChatFireworks", _call),
    "google_anthropic_vertex": (
        "langchain_google_vertexai.model_garden",
        "ChatAnthropicVertex",
        _call,
    ),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", _call),
    "google_vertexai": ("langchain_google_vertexai", "ChatVertexAI", _call),
    "groq": ("langchain_groq", "ChatGroq", _call),
    "huggingface": (
        "langchain_huggingface",
        "ChatHuggingFace",
        lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs),
    ),
    "ibm": (
        "langchain_ibm",
        "ChatWatsonx",
        lambda cls, model, **kwargs: cls(model_id=model, **kwargs),
    ),
    "mistralai": ("langchain_mistralai", "ChatMistralAI", _call),
    "nvidia": ("langchain_nvidia_ai_endpoints", "ChatNVIDIA", _call),
    "ollama": ("langchain_ollama", "ChatOllama", _call),
    "openai": ("langchain_openai", "ChatOpenAI", _call),
    "perplexity": ("langchain_perplexity", "ChatPerplexity", _call),
    "together": ("langchain_together", "ChatTogether", _call),
    "upstage": ("langchain_upstage", "ChatUpstage", _call),
    "xai": ("langchain_xai", "ChatXAI", _call),
}
"""Registry mapping provider names to their import configuration.

Each entry maps a provider key to a tuple of:

- `module_path`: The Python module path containing the chat model class.

    This may be a submodule (e.g., `'langchain_azure_ai.chat_models'`) if the class is
    not exported from the package root.
- `class_name`: The name of the chat model class to import.
- `creator_func`: A callable that instantiates the class with provided kwargs.

!!! note

    This dict is not exhaustive of all providers supported by LangChain, but is
    meant to cover the most popular ones and serve as a template for adding more
    providers in the future. If a provider is not in this dict, it can still be
    used with `init_chat_model` as long as its integration package is installed,
    but the provider key will not be inferred from the model name and must be
    specified explicitly via the `model_provider` parameter.

    Refer to the LangChain [integration documentation](https://docs.langchain.com/oss/python/integrations/providers/overview)
    for a full list of supported providers and their corresponding packages.
"""


def _import_module(module: str, class_name: str) -> ModuleType:
    """Import a module by name.

    Args:
        module: The fully qualified module name to import (e.g., `'langchain_openai'`).
        class_name: The name of the class being imported, used for error messages.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported, with a message suggesting
            the pip package to install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        # Extract package name from module path (e.g., "langchain_azure_ai.chat_models"
        # becomes "langchain-azure-ai")
        pkg = module.split(".", maxsplit=1)[0].replace("_", "-")
        msg = (
            f"Initializing {class_name} requires the {pkg} package. Please install it "
            f"with `pip install {pkg}`"
        )
        raise ImportError(msg) from e


@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_chat_model_creator(
    provider: str,
) -> Callable[..., BaseChatModel]:
    """Return a factory function that creates a chat model for the given provider.

    This function is cached to avoid repeated module imports.

    Args:
        provider: The name of the model provider (e.g., `'openai'`, `'anthropic'`).

            Must be a key in `_BUILTIN_PROVIDERS`.

    Returns:
        A callable that accepts model kwargs and returns a `BaseChatModel` instance for
            the specified provider.

    Raises:
        ValueError: If the provider is not in `_BUILTIN_PROVIDERS`.
        ImportError: If the provider's integration package is not installed.
    """
    if provider not in _BUILTIN_PROVIDERS:
        supported = ", ".join(_BUILTIN_PROVIDERS.keys())
        msg = f"Unsupported {provider=}.\n\nSupported model providers are: {supported}"
        raise ValueError(msg)

    pkg, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    try:
        module = _import_module(pkg, class_name)
    except ImportError as e:
        if provider != "ollama":
            raise
        # For backwards compatibility
        try:
            module = _import_module("langchain_community.chat_models", class_name)
        except ImportError:
            # If both langchain-ollama and langchain-community aren't available,
            # raise an error related to langchain-ollama
            raise e from None

    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)


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

    !!! note "Installation requirements"

        Requires the integration package for the chosen model provider to be installed.

        See the `model_provider` parameter below for specific package names
        (e.g., `pip install langchain-openai`).

        Refer to the [provider integration's API reference](https://docs.langchain.com/oss/python/integrations/providers)
        for supported model parameters to use as `**kwargs`.

    Args:
        model: The model name, optionally prefixed with provider (e.g., `'openai:gpt-4o'`).

            Prefer exact model IDs from provider docs over aliases for reliable behavior
            (e.g., dated versions like `'...-20250514'` instead of `'...-latest'`).

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
            - `solar...`                          -> `upstage`
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
            - `ibm`                     -> [`langchain-ibm`](https://docs.langchain.com/oss/python/integrations/providers/ibm)
            - `nvidia`                  -> [`langchain-nvidia-ai-endpoints`](https://docs.langchain.com/oss/python/integrations/providers/nvidia)
            - `xai`                     -> [`langchain-xai`](https://docs.langchain.com/oss/python/integrations/providers/xai)
            - `perplexity`              -> [`langchain-perplexity`](https://docs.langchain.com/oss/python/integrations/providers/perplexity)
            - `upstage`                 -> [`langchain-upstage`](https://docs.langchain.com/oss/python/integrations/providers/upstage)

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
    creator_func = _get_chat_model_creator(model_provider)
    return creator_func(model=model, **kwargs)


def _attempt_infer_model_provider(model_name: str) -> str | None:
    """Attempt to infer model provider from model name.

    Args:
        model_name: The name of the model to infer provider for.

    Returns:
        The inferred provider name, or `None` if no provider could be inferred.
    """
    model_lower = model_name.lower()

    # OpenAI models (including newer models and aliases)
    if any(
        model_lower.startswith(pre)
        for pre in (
            "gpt-",
            "o1",
            "o3",
            "chatgpt",
            "text-davinci",
        )
    ):
        return "openai"

    # Anthropic models
    if model_lower.startswith("claude"):
        return "anthropic"

    # Cohere models
    if model_lower.startswith("command"):
        return "cohere"

    # Fireworks models
    if model_lower.startswith("accounts/fireworks"):
        return "fireworks"

    # Google models
    if model_lower.startswith("gemini"):
        return "google_vertexai"

    # AWS Bedrock models
    if model_lower.startswith(("amazon.", "anthropic.", "meta.")):
        return "bedrock"

    # Mistral models
    if model_lower.startswith(("mistral", "mixtral")):
        return "mistralai"

    # DeepSeek models
    if model_lower.startswith("deepseek"):
        return "deepseek"

    # xAI models
    if model_lower.startswith("grok"):
        return "xai"

    # Perplexity models
    if model_lower.startswith("sonar"):
        return "perplexity"

    # Upstage models
    if model_lower.startswith("solar"):
        return "upstage"

    return None


def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    """Parse model name and provider, inferring provider if necessary."""
    # Handle provider:model format
    if (
        not model_provider
        and ":" in model
        and model.split(":", maxsplit=1)[0] in _BUILTIN_PROVIDERS
    ):
        model_provider = model.split(":", maxsplit=1)[0]
        model = ":".join(model.split(":")[1:])

    # Attempt to infer provider if not specified
    model_provider = model_provider or _attempt_infer_model_provider(model)

    if not model_provider:
        # Enhanced error message with suggestions
        supported_list = ", ".join(sorted(_BUILTIN_PROVIDERS))
        msg = (
            f"Unable to infer model provider for {model=}. "
            f"Please specify 'model_provider' directly.\n\n"
            f"Supported providers: {supported_list}\n\n"
            f"For help with specific providers, see: "
            f"https://docs.langchain.com/oss/python/integrations/providers"
        )
        raise ValueError(msg)

    # Normalize provider name
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider


def _remove_prefix(s: str, prefix: str) -> str:
    return s.removeprefix(prefix)


_DECLARATIVE_METHODS = ("bind_tools", "with_structured_output")


class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    def __init__(
        self,
        *,
        default_config: dict[str, Any] | None = None,
        configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[tuple[str, tuple[Any, ...], dict[str, Any]]] = (),
    ) -> None:
        self._default_config: dict[str, Any] = default_config or {}
        self._configurable_fields: Literal["any"] | list[str] = (
            "any" if configurable_fields == "any" else list(configurable_fields)
        )
        self._config_prefix = (
            config_prefix + "_"
            if config_prefix and not config_prefix.endswith("_")
            else config_prefix
        )
        self._queued_declarative_operations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = (
            list(
                queued_declarative_operations,
            )
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

    def _model(self, config: RunnableConfig | None = None) -> Runnable[Any, Any]:
        params = {**self._default_config, **self._model_params(config)}
        model = _init_chat_model_helper(**params)
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model

    def _model_params(self, config: RunnableConfig | None) -> dict[str, Any]:
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
        config = RunnableConfig(**(config or {}), **cast("RunnableConfig", kwargs))
        # Ensure config is not None after creation
        config = ensure_config(config)
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
    @override
    def InputType(self) -> TypeAlias:
        """Get the input type for this `Runnable`."""
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
    @override
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
    @override
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
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.__getattr__("bind_tools")(tools, **kwargs)

    # Explicitly added to satisfy downstream linters.
    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        return self.__getattr__("with_structured_output")(schema, **kwargs)
