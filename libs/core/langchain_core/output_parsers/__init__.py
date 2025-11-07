"""`OutputParser` classes parse the output of an LLM call into structured data.

!!! tip "Structured output"

    Output parsers emerged as an early solution to the challenge of obtaining structured
    output from LLMs.

    Today, most LLMs support [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs)
    natively. In such cases, using output parsers may be unnecessary, and you should
    leverage the model's built-in capabilities for structured output. Refer to the
    [documentation of your chosen model](https://docs.langchain.com/oss/python/integrations/providers/overview)
    for guidance on how to achieve structured output directly.

    Output parsers remain valuable when working with models that do not support
    structured output natively, or when you require additional processing or validation
    of the model's output beyond its inherent capabilities.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

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
    "BaseCumulativeTransformOutputParser",
    "BaseGenerationOutputParser",
    "BaseLLMOutputParser",
    "BaseOutputParser",
    "BaseTransformOutputParser",
    "CommaSeparatedListOutputParser",
    "JsonOutputKeyToolsParser",
    "JsonOutputParser",
    "JsonOutputToolsParser",
    "ListOutputParser",
    "MarkdownListOutputParser",
    "NumberedListOutputParser",
    "PydanticOutputParser",
    "PydanticToolsParser",
    "SimpleJsonOutputParser",
    "StrOutputParser",
    "XMLOutputParser",
]

_dynamic_imports = {
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
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return __all__
