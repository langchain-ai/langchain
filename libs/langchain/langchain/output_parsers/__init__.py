"""**OutputParser** classes parse the output of an LLM call.

**Class hierarchy:**

.. code-block::

    BaseLLMOutputParser --> BaseOutputParser --> <name>OutputParser  # ListOutputParser, PydanticOutputParser

**Main helpers:**

.. code-block::

    Serializable, Generation, PromptValue
"""  # noqa: E501

from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    ListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    PydanticOutputParser,
    XMLOutputParser,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain._api import create_importer
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.combining import CombiningOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain.output_parsers.enum import EnumOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.pandas_dataframe import PandasDataFrameOutputParser
from langchain.output_parsers.regex import RegexParser
from langchain.output_parsers.regex_dict import RegexDictParser
from langchain.output_parsers.retry import RetryOutputParser, RetryWithErrorOutputParser
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser
from langchain.output_parsers.yaml import YamlOutputParser

if TYPE_CHECKING:
    from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "GuardrailsOutputParser": "langchain_community.output_parsers.rail_parser",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BooleanOutputParser",
    "CombiningOutputParser",
    "CommaSeparatedListOutputParser",
    "DatetimeOutputParser",
    "EnumOutputParser",
    "GuardrailsOutputParser",
    "JsonOutputKeyToolsParser",
    "JsonOutputToolsParser",
    "ListOutputParser",
    "MarkdownListOutputParser",
    "NumberedListOutputParser",
    "OutputFixingParser",
    "PandasDataFrameOutputParser",
    "PydanticOutputParser",
    "PydanticToolsParser",
    "RegexDictParser",
    "RegexParser",
    "ResponseSchema",
    "RetryOutputParser",
    "RetryWithErrorOutputParser",
    "StructuredOutputParser",
    "XMLOutputParser",
    "YamlOutputParser",
]
