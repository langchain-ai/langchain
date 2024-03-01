"""**Language Model** is a type of model that can generate text or complete
text prompts. 

LangChain has two main classes to work with language models:
- **LLM** classes provide access to the large language model (**LLM**) APIs and services.
- **Chat Models** are a variation on language models.

**Class hierarchy:**

.. code-block::

    BaseLanguageModel --> BaseLLM --> LLM --> <name>  # Examples: AI21, HuggingFaceHub, OpenAI
                      --> BaseChatModel --> <name>    # Examples: ChatOpenAI, ChatGooglePalm

**Main helpers:**

.. code-block::

    LLMResult, PromptValue,
    CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun,
    CallbackManager, AsyncCallbackManager,
    AIMessage, BaseMessage, HumanMessage
"""  # noqa: E501

from langchain_core.language_models.base import (
    BaseLanguageModel,
    LanguageModelInput,
    LanguageModelLike,
    LanguageModelOutput,
    get_tokenizer,
)
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.language_models.llms import LLM, BaseLLM

__all__ = [
    "BaseLanguageModel",
    "BaseChatModel",
    "SimpleChatModel",
    "BaseLLM",
    "LLM",
    "LanguageModelInput",
    "get_tokenizer",
    "LanguageModelOutput",
    "LanguageModelLike",
]
