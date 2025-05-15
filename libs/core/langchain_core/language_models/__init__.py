"""Language models.

**Language Model** is a type of model that can generate text or complete
text prompts.

LangChain has two main classes to work with language models: **Chat Models**
and "old-fashioned" **LLMs**.

**Chat Models**

Language models that use a sequence of messages as inputs and return chat messages
as outputs (as opposed to using plain text). These are traditionally newer models (
older models are generally LLMs, see below). Chat models support the assignment of
distinct roles to conversation messages, helping to distinguish messages from the AI,
users, and instructions such as system messages.

The key abstraction for chat models is `BaseChatModel`. Implementations
should inherit from this class. Please see LangChain how-to guides with more
information on how to implement a custom chat model.

To implement a custom Chat Model, inherit from `BaseChatModel`. See
the following guide for more information on how to implement a custom Chat Model:

https://python.langchain.com/docs/how_to/custom_chat_model/

**LLMs**

Language models that takes a string as input and returns a string.
These are traditionally older models (newer models generally are Chat Models, see below).

Although the underlying models are string in, string out, the LangChain wrappers
also allow these models to take messages as input. This gives them the same interface
as Chat Models. When messages are passed in as input, they will be formatted into a
string under the hood before being passed to the underlying model.

To implement a custom LLM, inherit from `BaseLLM` or `LLM`.
Please see the following guide for more information on how to implement a custom LLM:

https://python.langchain.com/docs/how_to/custom_llm/


"""  # noqa: E501

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.language_models.base import (
        BaseLanguageModel,
        LangSmithParams,
        LanguageModelInput,
        LanguageModelLike,
        LanguageModelOutput,
        get_tokenizer,
    )
    from langchain_core.language_models.chat_models import (
        BaseChatModel,
        SimpleChatModel,
    )
    from langchain_core.language_models.fake import FakeListLLM, FakeStreamingListLLM
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
        FakeMessagesListChatModel,
        GenericFakeChatModel,
        ParrotFakeChatModel,
    )
    from langchain_core.language_models.llms import LLM, BaseLLM

__all__ = (
    "LLM",
    "BaseChatModel",
    "BaseLLM",
    "BaseLanguageModel",
    "FakeListChatModel",
    "FakeListLLM",
    "FakeMessagesListChatModel",
    "FakeStreamingListLLM",
    "GenericFakeChatModel",
    "LangSmithParams",
    "LanguageModelInput",
    "LanguageModelLike",
    "LanguageModelOutput",
    "ParrotFakeChatModel",
    "SimpleChatModel",
    "get_tokenizer",
)

_dynamic_imports = {
    "BaseLanguageModel": "base",
    "LangSmithParams": "base",
    "LanguageModelInput": "base",
    "LanguageModelLike": "base",
    "LanguageModelOutput": "base",
    "get_tokenizer": "base",
    "BaseChatModel": "chat_models",
    "SimpleChatModel": "chat_models",
    "FakeListLLM": "fake",
    "FakeStreamingListLLM": "fake",
    "FakeListChatModel": "fake_chat_models",
    "FakeMessagesListChatModel": "fake_chat_models",
    "GenericFakeChatModel": "fake_chat_models",
    "ParrotFakeChatModel": "fake_chat_models",
    "LLM": "llms",
    "BaseLLM": "llms",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
