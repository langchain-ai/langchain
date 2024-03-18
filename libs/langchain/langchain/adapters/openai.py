from typing import Any

DEPRECATED_IMPORTS = [
    "IndexableBaseModel",
    "Choice",
    "ChatCompletions",
    "ChoiceChunk",
    "ChatCompletionChunk",
    "convert_dict_to_message",
    "convert_message_to_dict",
    "convert_openai_messages",
    "ChatCompletion",
    "convert_messages_for_finetuning",
    "Completions",
    "Chat",
    "chat",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.adapters.openai import {name}`"
        )
    raise AttributeError()
