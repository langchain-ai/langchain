"""Module to test base parser implementations."""

from typing import Any, get_type_hints

import pytest
from typing_extensions import override

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseOutputParser,
    BaseTransformOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation


def test_asdict_replaces_deprecated_dict() -> None:
    class StrInvertCase(BaseTransformOutputParser[str]):
        def parse(self, text: str) -> str:
            return text.swapcase()

    parser = StrInvertCase()
    parser_dict = parser.asdict(exclude_none=True)
    assert parser_dict == {}
    with pytest.warns(LangChainDeprecationWarning, match="asdict"):
        assert parser.dict(exclude_none=True) == parser_dict


def test_base_output_parser_type_hints_resolve() -> None:
    assert get_type_hints(BaseOutputParser.asdict)["return"] == dict[str, Any]


def test_base_generation_parser() -> None:
    """Test Base Generation Output Parser."""

    class StrInvertCase(BaseGenerationOutputParser[str]):
        """An example parser that inverts the case of the characters in the message."""

        @override
        def parse_result(
            self, result: list[Generation], *, partial: bool = False
        ) -> str:
            """Parse a list of model Generations into a specific format.

            Args:
                result: A list of `Generation` to be parsed. The Generations are assumed
                    to be different candidate outputs for a single model input.
                    Many parsers assume that only a single generation is passed it in.
                    We will assert for that
                partial: Whether to allow partial results. This is used for parsers
                         that support streaming
            """
            if len(result) != 1:
                msg = "This output parser can only be used with a single generation."
                raise NotImplementedError(msg)
            generation = result[0]
            if not isinstance(generation, ChatGeneration):
                # Say that this one only works with chat generations
                msg = "This output parser can only be used with a chat generation."
                raise OutputParserException(msg)

            content = generation.message.content
            assert isinstance(content, str)
            return content.swapcase()

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hEllo")]))
    chain = model | StrInvertCase()
    assert chain.invoke("") == "HeLLO"


def test_base_transform_output_parser() -> None:
    """Test base transform output parser."""

    class StrInvertCase(BaseTransformOutputParser[str]):
        """An example parser that inverts the case of the characters in the message."""

        def parse(self, text: str) -> str:
            """Parse a single string into a specific format."""
            raise NotImplementedError

        @override
        def parse_result(
            self, result: list[Generation], *, partial: bool = False
        ) -> str:
            """Parse a list of model Generations into a specific format.

            Args:
                result: A list of `Generation` to be parsed. The Generations are assumed
                    to be different candidate outputs for a single model input.
                    Many parsers assume that only a single generation is passed it in.
                    We will assert for that
                partial: Whether to allow partial results. This is used for parsers
                         that support streaming
            """
            if len(result) != 1:
                msg = "This output parser can only be used with a single generation."
                raise NotImplementedError(msg)
            generation = result[0]
            if not isinstance(generation, ChatGeneration):
                # Say that this one only works with chat generations
                msg = "This output parser can only be used with a chat generation."
                raise OutputParserException(msg)
            content = generation.message.content
            assert isinstance(content, str)
            return content.swapcase()

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
    chain = model | StrInvertCase()
    # inputs to models are ignored, response is hard-coded in model definition
    chunks = list(chain.stream(""))
    assert chunks == ["HELLO", " ", "WORLD"]
