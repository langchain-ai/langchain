import warnings
from importlib import util
from typing import Any, Callable, Literal, Optional, Union, overload

from langchain_core._api import beta
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    SimpleChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable, RunnableLambda

__all__ = [
    "BaseChatModel",
    "SimpleChatModel",
    "generate_from_stream",
    "agenerate_from_stream",
    "init_chat_model",
]


def _runnable_support(initializer: Callable) -> Callable:
    @overload
    def wrapped(
        model: str, *, config_prefix: Literal[None] = None, **kwargs: Any
    ) -> BaseChatModel:
        ...

    @overload
    def wrapped(model: Optional[str], *, config_prefix: str, **kwargs: Any) -> Runnable:
        ...

    @overload
    def wrapped(
        model: Literal[None] = None,
        *,
        config_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable:
        ...

    def wrapped(
        model: Optional[str] = None,
        *,
        model_provider: Optional[str] = None,
        config_prefix: Optional[str] = None,
        configure_any: bool = False,
        **kwargs: Any,
    ) -> Union[BaseChatModel, Runnable]:
        if model and config_prefix is None:
            if configure_any:
                raise ValueError(
                    f"Must specify config_prefix if configure_any=True. Received "
                    f"{config_prefix=} and {configure_any=}."
                )
            return initializer(model, **kwargs)
        else:
            config_prefix = config_prefix + "_" if config_prefix is not None else ""
            if model:
                kwargs["model"] = model
            if model_provider:
                kwargs["model_provider"] = model_provider

            def from_config(
                input: LanguageModelInput, config: RunnableConfig
            ) -> BaseChatModel:
                config_params = {}
                if configure_any:
                    for k, v in config.get("configurable", {}).items():
                        if k.startswith(config_prefix):
                            config_params[_remove_prefix(k, config_prefix)] = v
                else:
                    for k, v in config.get("configurable", {}).items():
                        if k.startswith(config_prefix):
                            parsed = _remove_prefix(k, config_prefix)
                            if parsed in ("model", "model_provider"):
                                config_params[parsed] = v
                            elif config_prefix:
                                warnings.warn(
                                    f"Received unsupported config param to chat model. "
                                    f"Only config params {config_prefix + 'model'} and "
                                    f"{config_prefix + 'model_provider'} are supported."
                                    f" To support other params specify:"
                                    f" ``init_chat_model(..., configure_any=True)``."
                                )
                            else:
                                pass
                all_params = {**kwargs, **config_params}
                return initializer(**all_params)

            return RunnableLambda(from_config, name="configurable_chat_model")

    wrapped.__doc__ = initializer.__doc__
    return wrapped


# FOR CONTRIBUTORS: If adding support for a new provider, please append the provider
# name to the supported list in the docstring below. Do *not* change the order of the
# existing providers.
@beta()
@_runnable_support
def init_chat_model(
    model: str, *, model_provider: Optional[str] = None, **kwargs: Any
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
        config_prefix: If config_prefix is a non-empty string then model will be
            configurable at runtime via the
            ``config["configurable"]["{config_prefix}_model"]`` and
            ``config["configurable"]["{config_prefix}_model_provider"]`` keys. If
            config_prefix is an empty string or the model arg is None then model
            will be configurable via ``config["configurable"]["model"]``
            and ``config["configurable"]["model_provider"]`` keys. If config_prefix is
            None and model is not None then model will not be configurable.
        kwargs: Additional keyword args to pass to
            ``<<selected ChatModel>>.__init__(model=model_name, **kwargs)``.

    Returns:
        The BaseChatModel corresponding to the model_name and model_provider specified.

    Raises:
        ValueError: If model_provider cannot be inferred or isn't supported.
        ImportError: If the model provider integration package is not installed.

    Example:
        .. code-block:: python

            # pip install langchain langchain-openai langchain-anthropic langchain-google-vertexai
            from langchain.chat_models import init_chat_model

            gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
            claude_opus = init_chat_model("claude-3-opus-20240229", model_provider="anthropic", temperature=0)
            gemini_15 = init_chat_model("gemini-1.5-pro", model_provider="google_vertexai", temperature=0)

            gpt_4o.invoke("what's your name")
            claude_opus.invoke("what's your name")
            gemini_15.invoke("what's your name")


    Create a configurable model with no defaults:
        .. code-block:: python

            # pip install langchain langchain-openai langchain-anthropic
            from langchain.chat_models import init_chat_model

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

    Create a fully configurable model with defaults and a config prefix:
        .. code-block:: python

            # pip install langchain langchain-openai langchain-anthropic
            from langchain.chat_models import init_chat_model

            configurable_model_with_default = init_chat_model(
                "gpt-4o",
                model_provider="openai",
                config_prefix="foo",
                configure_any=True,  # this allows us to configure other params like temperature, max_tokens, etc at runtime.
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

    """  # noqa: E501
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

        return ChatAnthropic(model=model, **kwargs)
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

        return ChatMistralAI(model=model, **kwargs)
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
