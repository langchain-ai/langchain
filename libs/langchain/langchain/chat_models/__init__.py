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
            "Importing chat models from langchain is deprecated. Importing from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.chat_models import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(chat_models, name)

<<<<<<< HEAD
from langchain.chat_models.anthropic import ChatAnthropic
from langchain.chat_models.anyscale import ChatAnyscale
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chat_models.baichuan import ChatBaichuan
from langchain.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain.chat_models.bedrock import BedrockChat
from langchain.chat_models.cohere import ChatCohere
from langchain.chat_models.erniebot import ErnieBotChat
from langchain.chat_models.everlyai import ChatEverlyAI
from langchain.chat_models.fake import FakeListChatModel
from langchain.chat_models.fireworks import ChatFireworks
from langchain.chat_models.gigachat import GigaChat
from langchain.chat_models.google_palm import ChatGooglePalm
from langchain.chat_models.human import HumanInputChatModel
from langchain.chat_models.hunyuan import ChatHunyuan
from langchain.chat_models.javelin_ai_gateway import ChatJavelinAIGateway
from langchain.chat_models.jinachat import JinaChat
from langchain.chat_models.konko import ChatKonko
from langchain.chat_models.litellm import ChatLiteLLM
from langchain.chat_models.minimax import MiniMaxChat
from langchain.chat_models.mlflow_ai_gateway import ChatMLflowAIGateway
from langchain.chat_models.ollama import ChatOllama
from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models.pai_eas_endpoint import PaiEasChatEndpoint
from langchain.chat_models.promptlayer_openai import PromptLayerChatOpenAI
from langchain.chat_models.vertexai import ChatVertexAI
from langchain.chat_models.yandex import ChatYandexGPT
=======
>>>>>>> e438fe6be9f609b4229fe4c7688899425f758072

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
    "ChatJavelinAIGateway",
    "ChatKonko",
    "PaiEasChatEndpoint",
    "QianfanChatEndpoint",
    "ChatFireworks",
    "ChatYandexGPT",
    "ChatBaichuan",
    "ChatHunyuan",
    "GigaChat",
<<<<<<< HEAD
    "ErnieBotChat",
=======
    "VolcEngineMaasChat",
>>>>>>> e438fe6be9f609b4229fe4c7688899425f758072
]
