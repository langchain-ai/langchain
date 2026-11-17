"""Module to test base parser implementations."""

import asyncio
import threading
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


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("input", ["text", AIMessage(content="text")])
async def test_base_output_parser_ainvoke_uses_aparse(
    *, inherited: bool, input: str | AIMessage
) -> None:
    caller_loop = asyncio.get_running_loop()
    caller_thread = threading.get_ident()

    class AsyncParser(BaseOutputParser[str]):
        def parse(self, text: str) -> str:
            return asyncio.run(self.aparse(text))

        async def aparse(self, text: str) -> str:
            assert asyncio.get_running_loop() is caller_loop
            assert threading.get_ident() == caller_thread
            return text.upper()

    class InheritedParser(AsyncParser):
        pass

    parser = InheritedParser() if inherited else AsyncParser()
    assert await parser.ainvoke(input) == "TEXT"
    assert await parser.aparse_result([Generation(text="text")]) == "TEXT"


@pytest.mark.parametrize("override_aparse", [False, True])
@pytest.mark.parametrize("partial", [False, True])
async def test_base_output_parser_preserves_parse_result(
    *, override_aparse: bool, partial: bool
) -> None:
    caller_thread = threading.get_ident()
    message = AIMessage(
        content="", tool_calls=[{"name": "tool", "args": {}, "id": "call_1"}]
    )

    class ResultParser(BaseOutputParser[str]):
        def parse(self, text: str) -> str:
            pytest.fail(f"parse must not be called with {text!r}")

        def parse_result(
            self, result: list[Generation], *, partial: bool = False
        ) -> str:
            assert threading.get_ident() != caller_thread
            assert isinstance(result[0], ChatGeneration)
            assert result[0].message is message
            return f"result:{partial}"

    class AsyncResultParser(ResultParser):
        async def aparse(self, text: str) -> str:
            pytest.fail(f"aparse must not bypass parse_result for {text!r}")

    class InheritedParser(AsyncResultParser):
        pass

    parser = InheritedParser() if override_aparse else ResultParser()
    assert (
        await parser.aparse_result([ChatGeneration(message=message)], partial=partial)
        == f"result:{partial}"
    )
    assert await parser.ainvoke(message) == "result:False"


@pytest.mark.parametrize("inherited", [False, True])
async def test_base_output_parser_aparse_delegates_to_aparse_result(
    *, inherited: bool
) -> None:
    caller_thread = threading.get_ident()

    class DelegatingParser(BaseOutputParser[str]):
        def parse(self, text: str) -> str:
            assert threading.get_ident() != caller_thread
            return text.upper()

        async def aparse(self, text: str) -> str:
            return await self.aparse_result([Generation(text=text)])

    class InheritedParser(DelegatingParser):
        pass

    parser = InheritedParser() if inherited else DelegatingParser()
    assert await parser.ainvoke("text") == "TEXT"
    assert await parser.aparse("text") == "TEXT"
    assert await parser.aparse_result([Generation(text="text")]) == "TEXT"


async def test_base_output_parser_async_dispatch_is_task_local() -> None:
    class AsyncParser(BaseOutputParser[str]):
        def parse(self, text: str) -> str:
            pytest.fail(f"parse must not be called with {text!r}")

        async def aparse(self, text: str) -> str:
            if text == "error":
                msg = "parse failed"
                raise ValueError(msg)
            await asyncio.sleep(0)
            return text.upper()

    parser = AsyncParser()
    with pytest.raises(ValueError, match="parse failed"):
        await parser.aparse_result([Generation(text="error")])
    first, second = await asyncio.gather(parser.ainvoke("one"), parser.ainvoke("two"))
    assert (first, second) == ("ONE", "TWO")
    assert await parser.aparse_result([Generation(text="three")]) == "THREE"
    assert await parser.aparse_result([Generation(text="four")]) == "FOUR"


async def test_base_output_parser_keeps_sync_fallback() -> None:
    caller_thread = threading.get_ident()

    class SyncParser(BaseOutputParser[str]):
        def parse(self, text: str) -> str:
            assert threading.get_ident() != caller_thread
            return text.upper()

    parser = SyncParser()
    assert await parser.ainvoke("text") == "TEXT"
    assert await parser.aparse("text") == "TEXT"


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
