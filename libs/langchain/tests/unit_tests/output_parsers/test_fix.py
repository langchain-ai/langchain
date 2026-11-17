import asyncio
import threading
from datetime import UTC
from datetime import datetime as dt
from types import SimpleNamespace
from typing import Any, TypeVar
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from typing_extensions import override

from langchain_classic.output_parsers.boolean import BooleanOutputParser
from langchain_classic.output_parsers.datetime import DatetimeOutputParser
from langchain_classic.output_parsers.fix import OutputFixingParser
from langchain_classic.output_parsers.prompts import NAIVE_FIX_PROMPT

T = TypeVar("T")


class SuccessfulParseAfterRetries(BaseOutputParser[str]):
    parse_count: int = 0  # Number of times parse has been called
    attemp_count_before_success: int  # Number of times to fail before succeeding

    @override
    def parse(self, *args: Any, **kwargs: Any) -> str:
        self.parse_count += 1
        if self.parse_count <= self.attemp_count_before_success:
            msg = "error"
            raise OutputParserException(msg)
        return "parsed"


class SuccessfulParseAfterRetriesWithGetFormatInstructions(SuccessfulParseAfterRetries):
    def get_format_instructions(self) -> str:
        return "instructions"


@pytest.mark.parametrize(
    "base_parser",
    [
        SuccessfulParseAfterRetries(attemp_count_before_success=5),
        SuccessfulParseAfterRetriesWithGetFormatInstructions(
            attemp_count_before_success=5,
        ),
    ],
)
def test_output_fixing_parser_parse(
    base_parser: SuccessfulParseAfterRetries,
) -> None:
    # preparation
    n: int = base_parser.attemp_count_before_success  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = OutputFixingParser[str](
        parser=base_parser,
        max_retries=n,  # n times to retry, that is, (n+1) times call
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    # test
    assert parser.parse("completion") == "parsed"
    assert base_parser.parse_count == n + 1
    # TODO: test whether "instructions" is passed to the retry_chain


def test_output_fixing_parser_from_llm() -> None:
    def fake_llm(_: str) -> AIMessage:
        return AIMessage("2024-07-08T00:00:00.000000Z")

    llm = RunnableLambda(fake_llm)

    n = 1
    parser = OutputFixingParser.from_llm(
        llm=llm,
        parser=DatetimeOutputParser(),
        max_retries=n,
    )

    assert parser.parse("not a date")


@pytest.mark.parametrize(
    "base_parser",
    [
        SuccessfulParseAfterRetries(attemp_count_before_success=5),
        SuccessfulParseAfterRetriesWithGetFormatInstructions(
            attemp_count_before_success=5,
        ),
    ],
)
async def test_output_fixing_parser_aparse(
    base_parser: SuccessfulParseAfterRetries,
) -> None:
    n: int = base_parser.attemp_count_before_success  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = OutputFixingParser[str](
        parser=base_parser,
        max_retries=n,  # n times to retry, that is, (n+1) times call
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    assert (await parser.aparse("completion")) == "parsed"
    assert base_parser.parse_count == n + 1
    # TODO: test whether "instructions" is passed to the retry_chain


async def test_output_fixing_parser_ainvoke_sync_parser() -> None:
    caller_thread = threading.get_ident()

    def parse(completion: str) -> str:
        assert threading.get_ident() != caller_thread
        return completion.upper()

    parser = OutputFixingParser[str](
        parser=SimpleNamespace(parse=parse),
        retry_chain=RunnablePassthrough(),
    )
    assert await parser.ainvoke("completion") == "COMPLETION"


@pytest.mark.parametrize(
    ("method", "legacy", "has_instructions"),
    [
        ("run", True, True),
        ("invoke", True, True),
        ("invoke", False, True),
        ("invoke", False, False),
        ("arun", True, True),
        ("ainvoke", True, True),
        ("ainvoke", False, True),
    ],
)
async def test_output_fixing_parser_ainvoke_retry_dispatch(
    method: str, *, legacy: bool, has_instructions: bool
) -> None:
    caller_loop = asyncio.get_running_loop()
    caller_thread = threading.get_ident()
    parse_error = OutputParserException("bad completion")

    async def aparse(completion: str) -> str:
        assert asyncio.get_running_loop() is caller_loop
        if completion == "fixed":
            return "parsed"
        raise parse_error

    def retry(*_args: Any, **_kwargs: Any) -> str:
        assert threading.get_ident() != caller_thread
        return "fixed"

    async def aretry(*_args: Any, **_kwargs: Any) -> str:
        assert asyncio.get_running_loop() is caller_loop
        return "fixed"

    inner_parser = SimpleNamespace(aparse=aparse)
    if has_instructions:
        inner_parser.get_format_instructions = lambda: "instructions"
    retry_mock = (
        AsyncMock(side_effect=aretry)
        if method.startswith("a")
        else Mock(side_effect=retry)
    )
    retry_chain = SimpleNamespace(**{method: retry_mock})
    if method.startswith("a"):
        retry_chain.run = Mock()
        retry_chain.invoke = Mock()
    parser = OutputFixingParser[str](
        parser=inner_parser, retry_chain=retry_chain, legacy=legacy
    )
    assert await parser.ainvoke("bad") == "parsed"
    expected = {"completion": "bad", "error": repr(parse_error)}
    if has_instructions:
        expected["instructions"] = "instructions"
    args, kwargs = ((), expected) if method.endswith("run") else ((expected,), {})
    retry_mock.assert_called_once_with(*args, **kwargs)
    if isinstance(retry_mock, AsyncMock):
        retry_mock.assert_awaited_once_with(*args, **kwargs)
        retry_chain.run.assert_not_called()
        retry_chain.invoke.assert_not_called()


@pytest.mark.parametrize("method", ["parse", "aparse"])
@pytest.mark.parametrize("error_type", [OutputParserException, AttributeError])
async def test_output_fixing_parser_ainvoke_parser_failure(
    method: str, error_type: type[Exception]
) -> None:
    parse_error = error_type("bad completion")
    parse_mock = (
        AsyncMock(side_effect=parse_error)
        if method == "aparse"
        else Mock(side_effect=parse_error)
    )
    retry_chain = SimpleNamespace(run=Mock())
    parser = OutputFixingParser[str](
        parser=SimpleNamespace(**{method: parse_mock}),
        retry_chain=retry_chain,
        max_retries=0,
    )
    with pytest.raises(error_type) as exc_info:
        await parser.ainvoke("bad")
    assert exc_info.value is parse_error
    parse_mock.assert_called_once_with("bad")
    retry_chain.run.assert_not_called()


def test_output_fixing_parser_parse_fail() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = OutputFixingParser[str](
        parser=base_parser,
        max_retries=n - 1,  # n-1 times to retry, that is, n times call
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    with pytest.raises(OutputParserException):
        parser.parse("completion")
    assert base_parser.parse_count == n


async def test_output_fixing_parser_aparse_fail() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = OutputFixingParser[str](
        parser=base_parser,
        max_retries=n - 1,  # n-1 times to retry, that is, n times call
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    with pytest.raises(OutputParserException):
        await parser.aparse("completion")
    assert base_parser.parse_count == n


@pytest.mark.parametrize(
    "base_parser",
    [
        BooleanOutputParser(),
        DatetimeOutputParser(),
    ],
)
def test_output_fixing_parser_output_type(
    base_parser: BaseOutputParser,
) -> None:
    parser = OutputFixingParser[str](
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
    )
    assert parser.OutputType is base_parser.OutputType


@pytest.mark.parametrize(
    ("completion", "base_parser", "retry_chain", "expected"),
    [
        (
            "2024/07/08",
            DatetimeOutputParser(format="%Y-%m-%dT%H:%M:%S.%f%z"),
            NAIVE_FIX_PROMPT | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8, tzinfo=UTC),
        ),
        (
            # Case: retry_chain.InputType does not have 'instructions' key
            "2024/07/08",
            DatetimeOutputParser(format="%Y-%m-%dT%H:%M:%S.%f%z"),
            PromptTemplate.from_template("{completion}\n{error}")
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8, tzinfo=UTC),
        ),
    ],
)
def test_output_fixing_parser_parse_with_retry_chain(
    completion: str,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[dict[str, Any], str],
    expected: T,
) -> None:
    # NOTE: get_format_instructions of some parsers behave randomly
    instructions = base_parser.get_format_instructions()
    object.__setattr__(base_parser, "get_format_instructions", lambda: instructions)
    # test
    parser = OutputFixingParser[str](
        parser=base_parser,
        retry_chain=retry_chain,
        legacy=False,
    )
    assert parser.parse(completion) == expected


@pytest.mark.parametrize(
    ("completion", "base_parser", "retry_chain", "expected"),
    [
        (
            "2024/07/08",
            DatetimeOutputParser(format="%Y-%m-%dT%H:%M:%S.%f%z"),
            NAIVE_FIX_PROMPT | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8, tzinfo=UTC),
        ),
        (
            # Case: retry_chain.InputType does not have 'instructions' key
            "2024/07/08",
            DatetimeOutputParser(format="%Y-%m-%dT%H:%M:%S.%f%z"),
            PromptTemplate.from_template("{completion}\n{error}")
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8, tzinfo=UTC),
        ),
    ],
)
async def test_output_fixing_parser_aparse_with_retry_chain(
    completion: str,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[dict[str, Any], str],
    expected: T,
) -> None:
    instructions = base_parser.get_format_instructions()
    object.__setattr__(base_parser, "get_format_instructions", lambda: instructions)
    # test
    parser = OutputFixingParser[str](
        parser=base_parser,
        retry_chain=retry_chain,
        legacy=False,
    )
    assert (await parser.aparse(completion)) == expected
