import warnings

from langchain_core._api import LangChainDeprecationWarning

from langchain.utils.interactive_env import is_interactive_env


def __getattr__(name: str) -> None:
    # If not in interactive env, raise warning.
    from langchain_community.adapters import openai

    if not is_interactive_env():
        warnings.warn(
            "Importing from langchain is deprecated. Importing from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Instead of `from langchain.adapters.openai import {name}` "
            "Use `from langchain_community.adapters.openai import {name}`."
            "To install langchain-community run `pip install -U langchain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(openai, name)


__all__ = [  # noqa: F822
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
