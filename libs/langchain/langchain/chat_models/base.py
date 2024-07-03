from __future__ import annotations

import inspect
import warnings
from importlib import util
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

from langchain_core._api import beta
from langchain_core.language_models import (
    BaseChatModel,
    SimpleChatModel,
)
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.runnables import Runnable, RunnableConfig

__all__ = [
    "init_chat_model",
    # For backwards compatibility
    "BaseChatModel",
    "SimpleChatModel",
    "generate_from_stream",
    "agenerate_from_stream",
]


# FOR CONTRIBUTORS: If adding support for a new provider, please append the provider
# name to the supported list in the docstring below. Do *not* change the order of the
# existing providers.
@beta()
def init_chat_model(
    model: Optional[str] = None,
    *,
    model_provider: Optional[str] = None,
    configurable_fields: Optional[
        Union[Literal["any"], List[str], Tuple[str, ...]]
    ] = None,
    config_prefix: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Initialize a ChatModel from the model name and provider.

    Must have the integration package corresponding to the model provider installed.

    Args:
        model: The name of the model, e.g. "gpt-4o", "claude-3-opus-20240229".
        model_provider: The model provider. Supported model_provider values and the
            corresponding integration package:
                - openai (langchain-openai)
                - anthropic (langchain-anthropic)
                - azure_openai (langchain-openai)
                - google_vertexai (langchain-google-vertexai)
                - google_genai (langchain-google-genai)
                - bedrock (langchain-aws)
                - cohere (langchain-cohere)
                - fireworks (langchain-fireworks)
                - together (langchain-together)
                - mistralai (langchain-mistralai)
                - huggingface (langchain-huggingface)
                - groq (langchain-groq)
                - ollama (langchain-community)

            Will attempt to infer model_provider from model if not specified. The
            following providers will be inferred based on these model prefixes:
                - gpt-3... or gpt-4... -> openai
                - claude... -> anthropic
                - amazon.... -> bedrock
                - gemini... -> google_vertexai
                - command... -> cohere
                - accounts/fireworks... -> fireworks
        configurable_fields: Which model parameters are
            configurable:
                - None: No configurable fields.
                - "any": All fields are configurable. *See Security Note below.*
                - Union[List[str], Tuple[str, ...]]: Specified fields are configurable.

            Fields are assumed to have config_prefix stripped if there is a
            config_prefix. If model is specified, then defaults to None. If model is
            not specified, then defaults to ``("model", "model_provider")``.

            ***Security Note***: Setting ``configurable_fields="any"`` means fields like
            api_key, base_url, etc. can be altered at runtime, potentially redirecting
            model requests to a different service/user. Make sure that if you're
            accepting untrusted configurations that you enumerate the
            ``configurable_fields=(...)`` explicitly.

        config_prefix: If config_prefix is a non-empty string then model will be
            configurable at runtime via the
            ``config["configurable"]["{config_prefix}_{param}"]`` keys. If
            config_prefix is an empty string then model will be configurable via
            ``config["configurable"]["{param}"]``.
        kwargs: Additional keyword args to pass to
            ``<<selected ChatModel>>.__init__(model=model_name, **kwargs)``.

    Returns:
        A BaseChatModel corresponding to the model_name and model_provider specified if
        configurability is inferred to be False. If configurable, a chat model emulator
        that initializes the underlying model at runtime once a config is passed in.

    Raises:
        ValueError: If model_provider cannot be inferred or isn't supported.
        ImportError: If the model provider integration package is not installed.

    Initialize non-configurable models:
        .. code-block:: python

            # pip install langchain langchain-openai langchain-anthropic langchain-google-vertexai
            from langchain.chat_models import init_chat_model

            gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
            claude_opus = init_chat_model("claude-3-opus-20240229", model_provider="anthropic", temperature=0)
            gemini_15 = init_chat_model("gemini-1.5-pro", model_provider="google_vertexai", temperature=0)

            gpt_4o.invoke("what's your name")
            claude_opus.invoke("what's your name")
            gemini_15.invoke("what's your name")


    Create a partially configurable model with no default model:
        .. code-block:: python

            # pip install langchain langchain-openai langchain-anthropic
            from langchain.chat_models import init_chat_model

            # We don't need to specify configurable=True if a model isn't specified.
            configurable_model = init_chat_model(temperature=0)

            configurable_model.invoke(
                "what's your name",
                config={"configurable": {"model": "gpt-4o"}}
            )
            # GPT-4o response

            configurable_model.invoke(
                "what's your name",
                config={"configurable": {"model": "claude-3-5-sonnet-20240620"}}
            )
            # claude-3.5 sonnet response

    Create a fully configurable model with a default model and a config prefix:
        .. code-block:: python

            # pip install langchain langchain-openai langchain-anthropic
            from langchain.chat_models import init_chat_model

            configurable_model_with_default = init_chat_model(
                "gpt-4o",
                model_provider="openai",
                configurable_fields="any",  # this allows us to configure other params like temperature, max_tokens, etc at runtime.
                config_prefix="foo",
                temperature=0
            )

            configurable_model_with_default.invoke("what's your name")
            # GPT-4o response with temperature 0

            configurable_model_with_default.invoke(
                "what's your name",
                config={
                    "configurable": {
                        "foo_model": "claude-3-5-sonnet-20240620",
                        "foo_model_provider": "anthropic",
                        "foo_temperature": 0.6
                    }
                }
            )
            # Claude-3.5 sonnet response with temperature 0.6

    Bind tools to a configurable model:
        You can call any ChatModel declarative methods on a configurable model in the
        same way that you would with a normal model.

        .. code-block:: python

            # pip install langchain langchain-openai langchain-anthropic
            from langchain.chat_models import init_chat_model
            from langchain_core.pydantic_v1 import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            configurable_model = init_chat_model(
                "gpt-4o",
                configurable_fields=("model", "model_provider"),
                temperature=0
            )

            configurable_model_with_tools = configurable_model.bind_tools([GetWeather, GetPopulation])
            configurable_model_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            # GPT-4o response with tool calls

            configurable_model_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?",
                config={"configurable": {"model": "claude-3-5-sonnet-20240620"}}
            )
            # Claude-3.5 sonnet response with tools
    """  # noqa: E501
    if not model and not configurable_fields:
        configurable_fields = ("model", "model_provider")
    config_prefix = config_prefix or ""
    if config_prefix and not configurable_fields:
        warnings.warn(
            f"{config_prefix=} has been set but no fields are configurable. Set "
            f"`configurable_fields=(...)` to specify the model params that are "
            f"configurable."
        )

    if not configurable_fields:
        return _init_chat_model_helper(
            cast(str, model), model_provider=model_provider, **kwargs
        )
    else:
        if model:
            kwargs["model"] = model
        if model_provider:
            kwargs["model_provider"] = model_provider
        return cast(
            BaseChatModel,
            _ConfigurableModel(
                default_config=kwargs,
                config_prefix=config_prefix,
                configurable_fields=configurable_fields,
            ),
        )


def _init_chat_model_helper(
    model: str, *, model_provider: Optional[str] = None, **kwargs: Any
) -> BaseChatModel:
    model_provider = model_provider or _attempt_infer_model_provider(model)
    if not model_provider:
        raise ValueError(
            f"Unable to infer model provider for {model=}, please specify "
            f"model_provider directly."
        )
    model_provider = model_provider.replace("-", "_").lower()
    if model_provider == "openai":
        _check_pkg("langchain_openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, **kwargs)
    elif model_provider == "anthropic":
        _check_pkg("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, **kwargs)  # type: ignore[call-arg]
    elif model_provider == "azure_openai":
        _check_pkg("langchain_openai")
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(model=model, **kwargs)
    elif model_provider == "cohere":
        _check_pkg("langchain_cohere")
        from langchain_cohere import ChatCohere

        return ChatCohere(model=model, **kwargs)
    elif model_provider == "google_vertexai":
        _check_pkg("langchain_google_vertexai")
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(model=model, **kwargs)
    elif model_provider == "google_genai":
        _check_pkg("langchain_google_genai")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, **kwargs)
    elif model_provider == "fireworks":
        _check_pkg("langchain_fireworks")
        from langchain_fireworks import ChatFireworks

        return ChatFireworks(model=model, **kwargs)
    elif model_provider == "ollama":
        _check_pkg("langchain_community")
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=model, **kwargs)
    elif model_provider == "together":
        _check_pkg("langchain_together")
        from langchain_together import ChatTogether

        return ChatTogether(model=model, **kwargs)
    elif model_provider == "mistralai":
        _check_pkg("langchain_mistralai")
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=model, **kwargs)  # type: ignore[call-arg]
    elif model_provider == "huggingface":
        _check_pkg("langchain_huggingface")
        from langchain_huggingface import ChatHuggingFace

        return ChatHuggingFace(model_id=model, **kwargs)
    elif model_provider == "groq":
        _check_pkg("langchain_groq")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, **kwargs)
    elif model_provider == "bedrock":
        _check_pkg("langchain_aws")
        from langchain_aws import ChatBedrock

        # TODO: update to use model= once ChatBedrock supports
        return ChatBedrock(model_id=model, **kwargs)
    else:
        supported = ", ".join(_SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unsupported {model_provider=}.\n\nSupported model providers are: "
            f"{supported}"
        )


_SUPPORTED_PROVIDERS = {
    "openai",
    "anthropic",
    "azure_openai",
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
}


def _attempt_infer_model_provider(model_name: str) -> Optional[str]:
    if model_name.startswith("gpt-3") or model_name.startswith("gpt-4"):
        return "openai"
    elif model_name.startswith("claude"):
        return "anthropic"
    elif model_name.startswith("command"):
        return "cohere"
    elif model_name.startswith("accounts/fireworks"):
        return "fireworks"
    elif model_name.startswith("gemini"):
        return "google_vertexai"
    elif model_name.startswith("amazon."):
        return "bedrock"
    else:
        return None


def _check_pkg(pkg: str) -> None:
    if not util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        raise ImportError(
            f"Unable to import {pkg_kebab}. Please install with "
            f"`pip install -U {pkg_kebab}`"
        )


def _remove_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


_CHAT_MODEL_METHODS_TAKE_CONFIG = tuple(
    name
    for name, func in inspect.getmembers(BaseChatModel, inspect.isfunction)
    if "config" in inspect.signature(func).parameters
)

_DECLARATIVE_METHODS = (
    "assign",
    "bind",
    "map",
    "pick",
    "pipe",
    "with_alisteners",
    "with_fallbacks",
    "with_listeners",
    "with_retry",
    "with_types",
    "bind_tools",
    "with_structured_output",
    "configurable_fields",
    "configurable_alternatives",
)


class _ConfigurableModel:
    def __init__(
        self,
        *,
        default_config: Optional[dict] = None,
        configurable_fields: Union[Literal["any"], List[str], Tuple[str, ...]] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[Tuple[str, Tuple, Dict]] = (),
    ) -> None:
        self._default_config = default_config or {}
        self._configurable_fields = configurable_fields
        self._config_prefix = config_prefix + "_" if config_prefix else config_prefix
        self._queued_declarative_operations: List[Tuple[str, Tuple, Dict]] = list(
            queued_declarative_operations
        )

    def __getattr__(self, name: str) -> Any:
        if name in _CHAT_MODEL_METHODS_TAKE_CONFIG:

            def from_config(
                *args: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
            ) -> Any:
                return getattr(self._model(config), name)(
                    *args, config=config, **kwargs
                )

            return from_config
        elif name in _DECLARATIVE_METHODS:
            # Declarative operations cannot be applied until after an actual model
            # object is instantiated. So instead of returning the actual operation,
            # we record the operation and its arguments in a queue. This queue is
            # then applied in order whenever we actually instantiate the model (in
            # self._model()).
            def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
                self._queued_declarative_operations.append((name, args, kwargs))
                return self

            return queue
        elif self._default_config and (model := self._model()) and hasattr(model, name):
            return getattr(model, name)
        else:
            msg = f"{name} is not a BaseChatModel attribute"
            if self._default_config:
                msg += " and is not implemented on the default model"
            msg += "."
            raise AttributeError(msg)

    def _model(self, config: Optional[RunnableConfig] = None) -> Runnable:
        params = {**self._default_config, **self._model_params(config)}
        model = _init_chat_model_helper(**params)
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model

    def _model_params(self, config: Optional[RunnableConfig]) -> dict:
        config = config or {}
        model_params = {
            _remove_prefix(k, self._config_prefix): v
            for k, v in config.get("configurable", {}).items()
            if k.startswith(self._config_prefix)
        }
        if self._configurable_fields != "any":
            model_params = {
                k: v for k, v in model_params.items() if k in self._configurable_fields
            }
        return model_params

    def with_config(
        self,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> _ConfigurableModel:
        """Bind config to a Runnable, returning a new Runnable."""
        config = RunnableConfig(**(config or {}), **cast(RunnableConfig, kwargs))
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
                ("with_config", (), {"config": remaining_config})
            )
        return _ConfigurableModel(
            default_config={**self._default_config, **model_params},
            configurable_fields=self._configurable_fields,
            config_prefix=self._config_prefix,
            queued_declarative_operations=queued_declarative_operations,
        )
