"""**Prompt** is the input to the model.

Prompt is often constructed
from multiple components. Prompt classes and functions make constructing
 and working with prompts easy.

**Class hierarchy:**

.. code-block::

    BasePromptTemplate --> PipelinePromptTemplate
                           StringPromptTemplate --> PromptTemplate
                                                    FewShotPromptTemplate
                                                    FewShotPromptWithTemplates
                           BaseChatPromptTemplate --> AutoGPTPrompt
                                                      ChatPromptTemplate --> AgentScratchPadChatPromptTemplate



    BaseMessagePromptTemplate --> MessagesPlaceholder
                                  BaseStringMessagePromptTemplate --> ChatMessagePromptTemplate
                                                                      HumanMessagePromptTemplate
                                                                      AIMessagePromptTemplate
                                                                      SystemMessagePromptTemplate

    PromptValue --> StringPromptValue
                    ChatPromptValue

"""  # noqa: E501
from typing import Any

from langchain_core.example_selectors import (
    LengthBasedExampleSelector,
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    BasePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    FewShotPromptWithTemplates,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PipelinePromptTemplate,
    PromptTemplate,
    StringPromptTemplate,
    SystemMessagePromptTemplate,
    load_prompt,
)

from langchain.prompts.prompt import Prompt

DEPRECATED_IMPORTS = [
    "NGramOverlapExampleSelector",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.example_selectors import {name}`"
        )
    raise AttributeError()


__all__ = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "HumanMessagePromptTemplate",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "MessagesPlaceholder",
    "PipelinePromptTemplate",
    "PromptTemplate",
    "SemanticSimilarityExampleSelector",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
    "FewShotChatMessagePromptTemplate",
    "Prompt",
]
