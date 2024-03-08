"""**Adapters** are used to adapt LangChain models to other APIs.

LangChain integrates with many model providers.
While LangChain has its own message and model APIs,
LangChain has also made it as easy as
possible to explore other models by exposing an **adapter** to adapt LangChain
models to the other APIs, as to the OpenAI API.
"""

import importlib
from typing import Any

_module_lookup = {
    "Chat": "langchain_community.adapters.openai",
    "ChatCompletion": "langchain_community.adapters.openai",
    "ChatCompletionChunk": "langchain_community.adapters.openai",
    "ChatCompletions": "langchain_community.adapters.openai",
    "Choice": "langchain_community.adapters.openai",
    "ChoiceChunk": "langchain_community.adapters.openai",
    "Completions": "langchain_community.adapters.openai",
    "IndexableBaseModel": "langchain_community.adapters.openai",
    "aenumerate": "langchain_community.adapters.openai",
    "chat": "langchain_community.adapters.openai",
    "convert_dict_to_message": "langchain_community.adapters.openai",
    "convert_message_to_dict": "langchain_community.adapters.openai",
    "convert_messages_for_finetuning": "langchain_community.adapters.openai",
    "convert_openai_messages": "langchain_community.adapters.openai",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
