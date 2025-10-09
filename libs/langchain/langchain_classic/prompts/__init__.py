"""**Prompt** is the input to the model.

Prompt is often constructed
from multiple components. Prompt classes and functions make constructing and working
with prompts easy.
"""

from typing import TYPE_CHECKING, Any

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
    PromptTemplate,
    StringPromptTemplate,
    SystemMessagePromptTemplate,
    load_prompt,
)

from langchain_classic._api import create_importer
from langchain_classic.prompts.prompt import Prompt

if TYPE_CHECKING:
    from langchain_community.example_selectors.ngram_overlap import (
        NGramOverlapExampleSelector,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
MODULE_LOOKUP = {
    "NGramOverlapExampleSelector": (
        "langchain_community.example_selectors.ngram_overlap"
    ),
}

_import_attribute = create_importer(__file__, module_lookup=MODULE_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "FewShotChatMessagePromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "HumanMessagePromptTemplate",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "MessagesPlaceholder",
    "NGramOverlapExampleSelector",
    "Prompt",
    "PromptTemplate",
    "SemanticSimilarityExampleSelector",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
]
