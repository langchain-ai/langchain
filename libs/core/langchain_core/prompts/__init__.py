"""**Prompt** is the input to the model.

Prompt is often constructed
from multiple components and prompt values. Prompt classes and functions make constructing
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

"""  # noqa: E501

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

if TYPE_CHECKING:
    from langchain_core.prompts.base import (
        BasePromptTemplate,
        aformat_document,
        format_document,
    )
    from langchain_core.prompts.chat import (
        AIMessagePromptTemplate,
        BaseChatPromptTemplate,
        ChatMessagePromptTemplate,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
    )
    from langchain_core.prompts.few_shot import (
        FewShotChatMessagePromptTemplate,
        FewShotPromptTemplate,
    )
    from langchain_core.prompts.few_shot_with_templates import (
        FewShotPromptWithTemplates,
    )
    from langchain_core.prompts.loading import load_prompt
    from langchain_core.prompts.pipeline import PipelinePromptTemplate
    from langchain_core.prompts.prompt import PromptTemplate
    from langchain_core.prompts.string import (
        StringPromptTemplate,
        check_valid_template,
        get_template_variables,
        jinja2_formatter,
        validate_jinja2,
    )

__all__ = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "FewShotChatMessagePromptTemplate",
    "HumanMessagePromptTemplate",
    "MessagesPlaceholder",
    "PipelinePromptTemplate",
    "PromptTemplate",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
    "format_document",
    "aformat_document",
    "check_valid_template",
    "get_template_variables",
    "jinja2_formatter",
    "validate_jinja2",
]

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="prompts",
    dynamic_imports={
        "BasePromptTemplate": "base",
        "format_document": "base",
        "aformat_document": "base",
        "AIMessagePromptTemplate": "chat",
        "BaseChatPromptTemplate": "chat",
        "ChatMessagePromptTemplate": "chat",
        "ChatPromptTemplate": "chat",
        "HumanMessagePromptTemplate": "chat",
        "MessagesPlaceholder": "chat",
        "SystemMessagePromptTemplate": "chat",
        "FewShotChatMessagePromptTemplate": "few_shot",
        "FewShotPromptTemplate": "few_shot",
        "FewShotPromptWithTemplates": "few_shot_with_templates",
        "load_prompt": "loading",
        "PipelinePromptTemplate": "pipeline",
        "PromptTemplate": "prompt",
        "StringPromptTemplate": "string",
        "check_valid_template": "string",
        "get_template_variables": "string",
        "jinja2_formatter": "string",
        "validate_jinja2": "string",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
