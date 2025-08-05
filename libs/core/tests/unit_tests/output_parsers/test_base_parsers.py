"""Module to test base parser implementations."""

from typing import Union

from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModelV1
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseTransformOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.v1.messages import AIMessage as AIMessageV1


def test_base_generation_parser() -> None:
    """Test Base Generation Output Parser."""

    class StrInvertCase(BaseGenerationOutputParser[str]):
        """An example parser that inverts the case of the characters in the message."""

        @override
        def parse_result(
            self, result: Union[list[Generation], AIMessageV1], *, partial: bool = False
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
            if isinstance(result, AIMessageV1):
                content = result.text
            else:
                if len(result) != 1:
                    msg = (
                        "This output parser can only be used with a single generation."
                    )
                    raise NotImplementedError(msg)
                generation = result[0]
                if not isinstance(generation, ChatGeneration):
                    # Say that this one only works with chat generations
                    msg = "This output parser can only be used with a chat generation."
                    raise OutputParserException(msg)
                assert isinstance(generation.message.content, str)
                content = generation.message.content

            assert isinstance(content, str)
            return content.swapcase()

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hEllo")]))
    chain = model | StrInvertCase()
    assert chain.invoke("") == "HeLLO"

    model_v1 = GenericFakeChatModelV1(messages=iter([AIMessageV1("hEllo")]))
    chain_v1 = model_v1 | StrInvertCase()
    assert chain_v1.invoke("") == "HeLLO"


def test_base_transform_output_parser() -> None:
    """Test base transform output parser."""

    class StrInvertCase(BaseTransformOutputParser[str]):
        """An example parser that inverts the case of the characters in the message."""

        def parse(self, text: str) -> str:
            """Parse a single string into a specific format."""
            raise NotImplementedError

        @override
        def parse_result(
            self, result: Union[list[Generation], AIMessageV1], *, partial: bool = False
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
            if isinstance(result, AIMessageV1):
                content = result.text
            else:
                if len(result) != 1:
                    msg = (
                        "This output parser can only be used with a single generation."
                    )
                    raise NotImplementedError(msg)
                generation = result[0]
                if not isinstance(generation, ChatGeneration):
                    # Say that this one only works with chat generations
                    msg = "This output parser can only be used with a chat generation."
                    raise OutputParserException(msg)
                assert isinstance(generation.message.content, str)
                content = generation.message.content

            assert isinstance(content, str)
            return content.swapcase()

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
    chain = model | StrInvertCase()
    # inputs to models are ignored, response is hard-coded in model definition
    chunks = list(chain.stream(""))
    assert chunks == ["HELLO", " ", "WORLD"]

    model_v1 = GenericFakeChatModelV1(message_chunks=["hello", " ", "world"])
    chain_v1 = model_v1 | StrInvertCase()
    chunks = list(chain_v1.stream(""))
    assert chunks == ["HELLO", " ", "WORLD", ""]
