"""**OutputParser** classes parse the output of an LLM call.

**Class hierarchy:**

.. code-block::

    BaseLLMOutputParser --> BaseOutputParser --> <name>OutputParser  # ListOutputParser, PydanticOutputParser

**Main helpers:**

.. code-block::

    Serializable, Generation, PromptValue
"""  # noqa: E501

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

if TYPE_CHECKING:
    from langchain_core.output_parsers.base import (
        BaseGenerationOutputParser,
        BaseLLMOutputParser,
        BaseOutputParser,
    )
    from langchain_core.output_parsers.json import (
        JsonOutputParser,
        SimpleJsonOutputParser,
    )
    from langchain_core.output_parsers.list import (
        CommaSeparatedListOutputParser,
        ListOutputParser,
        MarkdownListOutputParser,
        NumberedListOutputParser,
    )
    from langchain_core.output_parsers.openai_tools import (
        JsonOutputKeyToolsParser,
        JsonOutputToolsParser,
        PydanticToolsParser,
    )
    from langchain_core.output_parsers.pydantic import PydanticOutputParser
    from langchain_core.output_parsers.string import StrOutputParser
    from langchain_core.output_parsers.transform import (
        BaseCumulativeTransformOutputParser,
        BaseTransformOutputParser,
    )
    from langchain_core.output_parsers.xml import XMLOutputParser

__all__ = [
    "BaseLLMOutputParser",
    "BaseGenerationOutputParser",
    "BaseOutputParser",
    "ListOutputParser",
    "CommaSeparatedListOutputParser",
    "NumberedListOutputParser",
    "MarkdownListOutputParser",
    "StrOutputParser",
    "BaseTransformOutputParser",
    "BaseCumulativeTransformOutputParser",
    "SimpleJsonOutputParser",
    "XMLOutputParser",
    "JsonOutputParser",
    "PydanticOutputParser",
    "JsonOutputToolsParser",
    "JsonOutputKeyToolsParser",
    "PydanticToolsParser",
]

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="output_parsers",
    dynamic_imports={
        "BaseLLMOutputParser": "base",
        "BaseGenerationOutputParser": "base",
        "BaseOutputParser": "base",
        "JsonOutputParser": "json",
        "SimpleJsonOutputParser": "json",
        "ListOutputParser": "list",
        "CommaSeparatedListOutputParser": "list",
        "MarkdownListOutputParser": "list",
        "NumberedListOutputParser": "list",
        "JsonOutputKeyToolsParser": "openai_tools",
        "JsonOutputToolsParser": "openai_tools",
        "PydanticToolsParser": "openai_tools",
        "PydanticOutputParser": "pydantic",
        "StrOutputParser": "string",
        "BaseTransformOutputParser": "transform",
        "BaseCumulativeTransformOutputParser": "transform",
        "XMLOutputParser": "xml",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
