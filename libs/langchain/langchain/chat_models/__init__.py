"""**Chat Models** are a variation on language models.

While Chat Models use language models under the hood, the interface they expose
is a bit different. Rather than expose a "text in, text out" API, they expose
an interface where "chat messages" are the inputs and outputs.

**Class hierarchy:**

.. code-block::

    BaseLanguageModel --> BaseChatModel --> <name>  # Examples: ChatOpenAI, ChatGooglePalm

**Main helpers:**

.. code-block::

    AIMessage, BaseMessage, HumanMessage
"""  # noqa: E501
import warnings

from langchain_core._api import LangChainDeprecationWarning

from langchain.utils.interactive_env import is_interactive_env


def __getattr__(name: str) -> None:
    from langchain_community import chat_models

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing chat models from gigachain is deprecated. Importing from "
            "gigachain will no longer be supported as of gigachain==0.2.0. "
            "Please import from gigachain-community instead:\n\n"
            f"`from langchain_community.chat_models import {name}`.\n\n"
            "To install gigachain-community run `pip install -U gigachain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(chat_models, name)


__all__ = [
    "ChatOpenAI",
    "BedrockChat",
    "AzureChatOpenAI",
    "FakeListChatModel",
    "PromptLayerChatOpenAI",
    "ChatDatabricks",
    "ChatEverlyAI",
    "ChatAnthropic",
    "ChatCohere",
    "ChatGooglePalm",
    "ChatMlflow",
    "ChatMLflowAIGateway",
    "ChatOllama",
    "ChatVertexAI",
    "JinaChat",
    "HumanInputChatModel",
    "MiniMaxChat",
    "ChatAnyscale",
    "ChatLiteLLM",
    "ErnieBotChat",
    "GigaChat",
    "ChatJavelinAIGateway",
    "ChatKonko",
    "PaiEasChatEndpoint",
    "QianfanChatEndpoint",
    "ChatFireworks",
    "ChatYandexGPT",
    "ChatBaichuan",
    "ChatHunyuan",
    "GigaChat",
    "VolcEngineMaasChat",
]
