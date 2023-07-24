from typing import Dict, Type

from langchain.chat_models.anthropic import ChatAnthropic
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.fake import FakeListChatModel
from langchain.chat_models.google_palm import ChatGooglePalm
from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models.promptlayer_openai import PromptLayerChatOpenAI
from langchain.chat_models.vertexai import ChatVertexAI

__all__ = [
    "ChatOpenAI",
    "AzureChatOpenAI",
    "FakeListChatModel",
    "PromptLayerChatOpenAI",
    "ChatAnthropic",
    "ChatGooglePalm",
    "ChatVertexAI",
]

type_to_cls_dict: Dict[str, Type[BaseChatModel]] = {
    "azure-openai-chat": AzureChatOpenAI,
    "openai-chat": ChatOpenAI,
}
