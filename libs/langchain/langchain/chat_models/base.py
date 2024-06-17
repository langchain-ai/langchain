from importlib import util
from typing import Any, Optional

from langchain_core._api import beta
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    SimpleChatModel,
    agenerate_from_stream,
    generate_from_stream,
)

__all__ = [
    "BaseChatModel",
    "SimpleChatModel",
    "generate_from_stream",
    "agenerate_from_stream",
    "init_chat_model",
]


# FOR CONTRIBUTORS: If adding support for a new provider, please append the provider
# name to the supported list in the docstring below. Do *not* change the order of the
# existing providers.
@beta()
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
        kwargs: Additional keyword args to pass to
            ``<<selected ChatModel>>.__init__(model=model_name, **kwargs)``.

    Returns:
        The BaseChatModel corresponding to the model_name and model_provider specified.

    Raises:
        ValueError: If model_provider cannot be inferred or isn't supported.
        ImportError: If the model provider integration package is not installed.

    Example:
        .. code-block:: python

            from langchain.chat_models import init_chat_model

            gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
            claude_opus = init_chat_model("claude-3-opus-20240229", model_provider="anthropic", temperature=0)
            gemini_15 = init_chat_model("gemini-1.5-pro", model_provider="google_vertexai", temperature=0)

            gpt_4o.invoke("what's your name")
            claude_opus.invoke("what's your name")
            gemini_15.invoke("what's your name")
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
