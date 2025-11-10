"""Entrypoint to using [chat models](https://docs.langchain.com/oss/python/langchain/models) in LangChain."""  # noqa: E501

from langchain_core.language_models import BaseChatModel

from langchain.chat_models.base import init_chat_model

__all__ = ["BaseChatModel", "init_chat_model"]
