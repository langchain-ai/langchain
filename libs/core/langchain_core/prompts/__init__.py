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
from langchain_core.prompts.base import BasePromptTemplate, format_document
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    ChatPromptValue,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.few_shot import (
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain_core.prompts.loading import load_prompt
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import Prompt, PromptTemplate
from langchain_core.prompts.str import StringPromptTemplate, StringPromptValue

__all__ = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "ChatPromptValue",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "FewShotChatMessagePromptTemplate",
    "HumanMessagePromptTemplate",
    "MessagesPlaceholder",
    "PipelinePromptTemplate",
    "Prompt",
    "PromptTemplate",
    "PromptValue",
    "StringPromptValue",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
    "format_document",
]

from langchain_core.prompts.value import PromptValue
