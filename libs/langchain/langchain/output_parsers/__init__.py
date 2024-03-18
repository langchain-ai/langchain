"""**OutputParser** classes parse the output of an LLM call.

**Class hierarchy:**

.. code-block::

    BaseLLMOutputParser --> BaseOutputParser --> <name>OutputParser  # ListOutputParser, PydanticOutputParser

**Main helpers:**

.. code-block::

    Serializable, Generation, PromptValue
"""  # noqa: E501
from typing import Any

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.combining import CombiningOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain.output_parsers.enum import EnumOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.list import (
    CommaSeparatedListOutputParser,
    ListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
)
from langchain.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain.output_parsers.pandas_dataframe import PandasDataFrameOutputParser
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.output_parsers.regex import RegexParser
from langchain.output_parsers.regex_dict import RegexDictParser
from langchain.output_parsers.retry import RetryOutputParser, RetryWithErrorOutputParser
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser
from langchain.output_parsers.xml import XMLOutputParser
from langchain.output_parsers.yaml import YamlOutputParser

DEPRECATED_IMPORTS = [
    "GuardrailsOutputParser",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.output_parsers import {name}`"
        )

    raise AttributeError()


__all__ = [
    "BooleanOutputParser",
    "CombiningOutputParser",
    "CommaSeparatedListOutputParser",
    "DatetimeOutputParser",
    "EnumOutputParser",
    "ListOutputParser",
    "MarkdownListOutputParser",
    "NumberedListOutputParser",
    "OutputFixingParser",
    "PandasDataFrameOutputParser",
    "PydanticOutputParser",
    "RegexDictParser",
    "RegexParser",
    "ResponseSchema",
    "RetryOutputParser",
    "RetryWithErrorOutputParser",
    "StructuredOutputParser",
    "XMLOutputParser",
    "JsonOutputToolsParser",
    "PydanticToolsParser",
    "JsonOutputKeyToolsParser",
    "YamlOutputParser",
]
