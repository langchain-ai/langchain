"""Microsoft Prompty integration for LangChain."""

from langchain_prompty.core import InvokerFactory
from langchain_prompty.langchain import create_chat_prompt
from langchain_prompty.parsers import PromptyChatParser
from langchain_prompty.renderers import MustacheRenderer

InvokerFactory().register_renderer("mustache", MustacheRenderer)
InvokerFactory().register_parser("prompty.chat", PromptyChatParser)

__all__ = ["create_chat_prompt"]
