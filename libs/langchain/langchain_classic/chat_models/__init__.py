"""**Chat Models** are a variation on language models.

While Chat Models use language models under the hood, the interface they expose
is a bit different. Rather than expose a "text in, text out" API, they expose
an interface where "chat messages" are the inputs and outputs.
"""

import warnings

from langchain_core._api import LangChainDeprecationWarning

from langchain_classic._api.interactive_env import is_interactive_env
from langchain_classic.chat_models.base import init_chat_model


def __getattr__(name: str) -> None:
    from langchain_community import chat_models

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing chat models from langchain is deprecated. Importing from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.chat_models import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            stacklevel=2,
            category=LangChainDeprecationWarning,
        )

    return getattr(chat_models, name)


__all__ = [
    "AzureChatOpenAI",
    "BedrockChat",
    "ChatAnthropic",
    "ChatAnyscale",
    "ChatBaichuan",
    "ChatCohere",
    "ChatDatabricks",
    "ChatEverlyAI",
    "ChatFireworks",
    "ChatGooglePalm",
    "ChatHunyuan",
    "ChatJavelinAIGateway",
    "ChatKonko",
    "ChatLiteLLM",
    "ChatMLflowAIGateway",
    "ChatMlflow",
    "ChatOllama",
    "ChatOpenAI",
    "ChatVertexAI",
    "ChatYandexGPT",
    "ErnieBotChat",
    "FakeListChatModel",
    "GigaChat",
    "HumanInputChatModel",
    "JinaChat",
    "MiniMaxChat",
    "PaiEasChatEndpoint",
    "PromptLayerChatOpenAI",
    "QianfanChatEndpoint",
    "VolcEngineMaasChat",
    "init_chat_model",
]
