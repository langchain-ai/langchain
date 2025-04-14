"""**OutputParser** classes parse the output of an LLM call.

**Class hierarchy:**

.. code-block::

    BaseLLMOutputParser --> BaseOutputParser --> <name>OutputParser  # ListOutputParser, PydanticOutputParser

**Main helpers:**

.. code-block::

    Serializable, Generation, PromptValue
"""  # noqa: E501

from importlib import import_module
from typing import TYPE_CHECKING

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
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
