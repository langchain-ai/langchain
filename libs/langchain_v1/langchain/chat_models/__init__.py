"""Entrypoint to using [Chat Models](https://docs.langchain.com/oss/python/langchain/models) in LangChain.

See [the docs](https://docs.langchain.com/oss/python/langchain/models) for more
details.
"""  # noqa: E501

from langchain_core.language_models import BaseChatModel

from langchain.chat_models.base import init_chat_model

__all__ = ["BaseChatModel", "init_chat_model"]
