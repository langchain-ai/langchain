from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.adapters.openai import (
        Chat,
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletions,
        Choice,
        ChoiceChunk,
        Completions,
        IndexableBaseModel,
        chat,
        convert_dict_to_message,
        convert_message_to_dict,
        convert_messages_for_finetuning,
        convert_openai_messages,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
MODULE_LOOKUP = {
    "IndexableBaseModel": "langchain_community.adapters.openai",
    "Choice": "langchain_community.adapters.openai",
    "ChatCompletions": "langchain_community.adapters.openai",
    "ChoiceChunk": "langchain_community.adapters.openai",
    "ChatCompletionChunk": "langchain_community.adapters.openai",
    "convert_dict_to_message": "langchain_community.adapters.openai",
    "convert_message_to_dict": "langchain_community.adapters.openai",
    "convert_openai_messages": "langchain_community.adapters.openai",
    "ChatCompletion": "langchain_community.adapters.openai",
    "convert_messages_for_finetuning": "langchain_community.adapters.openai",
    "Completions": "langchain_community.adapters.openai",
    "Chat": "langchain_community.adapters.openai",
    "chat": "langchain_community.adapters.openai",
}

_import_attribute = create_importer(__file__, deprecated_lookups=MODULE_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
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
