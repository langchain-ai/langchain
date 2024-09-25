"""Module to test base parser implementations."""

from typing import Optional as Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseTransformOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation


def test_base_generation_parser() -> None:
    """Test Base Generation Output Parser."""

    class StrInvertCase(BaseGenerationOutputParser[str]):
        """An example parser that inverts the case of the characters in the message."""

        def parse_result(
            self, result: list[Generation], *, partial: bool = False
        ) -> str:
            """Parse a list of model Generations into a specific format.

            Args:
                result: A list of Generations to be parsed. The Generations are assumed
                    to be different candidate outputs for a single model input.
                    Many parsers assume that only a single generation is passed it in.
                    We will assert for that
                partial: Whether to allow partial results. This is used for parsers
                         that support streaming
            """
            if len(result) != 1:
                raise NotImplementedError(
                    "This output parser can only be used with a single generation."
                )
            generation = result[0]
            if not isinstance(generation, ChatGeneration):
                # Say that this one only works with chat generations
                raise OutputParserException(
                    "This output parser can only be used with a chat generation."
                )

            content = generation.message.content
            assert isinstance(content, str)
            return content.swapcase()  # type: ignore

    StrInvertCase.model_rebuild()

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hEllo")]))
    chain = model | StrInvertCase()
    assert chain.invoke("") == "HeLLO"


def test_base_transform_output_parser() -> None:
    """Test base transform output parser."""

    class StrInvertCase(BaseTransformOutputParser[str]):
        """An example parser that inverts the case of the characters in the message."""

        def parse(self, text: str) -> str:
            """Parse a single string into a specific format."""
            raise NotImplementedError()

        def parse_result(
            self, result: list[Generation], *, partial: bool = False
        ) -> str:
            """Parse a list of model Generations into a specific format.

            Args:
                result: A list of Generations to be parsed. The Generations are assumed
                    to be different candidate outputs for a single model input.
                    Many parsers assume that only a single generation is passed it in.
                    We will assert for that
                partial: Whether to allow partial results. This is used for parsers
                         that support streaming
            """
            if len(result) != 1:
                raise NotImplementedError(
                    "This output parser can only be used with a single generation."
                )
            generation = result[0]
            if not isinstance(generation, ChatGeneration):
                # Say that this one only works with chat generations
                raise OutputParserException(
                    "This output parser can only be used with a chat generation."
                )
            content = generation.message.content
            assert isinstance(content, str)
            return content.swapcase()  # type: ignore

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
    chain = model | StrInvertCase()
    # inputs to models are ignored, response is hard-coded in model definition
    chunks = [chunk for chunk in chain.stream("")]
    assert chunks == ["HELLO", " ", "WORLD"]
