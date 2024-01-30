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

from langchain_community.chat_models.anthropic import ChatAnthropic
from langchain_community.chat_models.anyscale import ChatAnyscale
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.chat_models.baichuan import ChatBaichuan
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_community.chat_models.cohere import ChatCohere
from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_community.chat_models.ernie import ErnieBotChat
from langchain_community.chat_models.everlyai import ChatEverlyAI
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.chat_models.google_palm import ChatGooglePalm
from langchain_community.chat_models.gpt_router import GPTRouter
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.chat_models.human import HumanInputChatModel
from langchain_community.chat_models.hunyuan import ChatHunyuan
from langchain_community.chat_models.javelin_ai_gateway import ChatJavelinAIGateway
from langchain_community.chat_models.jinachat import JinaChat
from langchain_community.chat_models.konko import ChatKonko
from langchain_community.chat_models.litellm import ChatLiteLLM
from langchain_community.chat_models.llama_edge import LlamaEdgeChatService
from langchain_community.chat_models.minimax import MiniMaxChat
from langchain_community.chat_models.mlflow import ChatMlflow
from langchain_community.chat_models.mlflow_ai_gateway import ChatMLflowAIGateway
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.chat_models.pai_eas_endpoint import PaiEasChatEndpoint
from langchain_community.chat_models.promptlayer_openai import PromptLayerChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.chat_models.volcengine_maas import VolcEngineMaasChat
from langchain_community.chat_models.yandex import ChatYandexGPT
from langchain_community.chat_models.zhipuai import ChatZhipuAI

__all__ = [
    "LlamaEdgeChatService",
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
    "ChatHuggingFace",
    "HumanInputChatModel",
    "MiniMaxChat",
    "ChatAnyscale",
    "ChatLiteLLM",
    "ErnieBotChat",
    "ChatJavelinAIGateway",
    "ChatKonko",
    "PaiEasChatEndpoint",
    "QianfanChatEndpoint",
    "ChatTongyi",
    "ChatFireworks",
    "ChatYandexGPT",
    "ChatBaichuan",
    "ChatHunyuan",
    "GigaChat",
    "VolcEngineMaasChat",
    "GPTRouter",
    "ChatZhipuAI",
]
