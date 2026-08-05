"""Claude (Anthropic) partner package for LangChain."""

from langchain_anthropic._version import __version__
from langchain_anthropic.chat_models import (
    ChatAnthropic,
    UnsupportedToolSchemaWarning,
    convert_to_anthropic_tool,
)
from langchain_anthropic.llms import AnthropicLLM

__all__ = [
    "AnthropicLLM",
    "ChatAnthropic",
    "UnsupportedToolSchemaWarning",
    "__version__",
    "convert_to_anthropic_tool",
]
