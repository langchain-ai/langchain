from datetime import datetime as dt
from typing import Any, Callable, Dict, Optional, TypeVar

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain.output_parsers.fix import BaseOutputParser, OutputFixingParser
from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT

T = TypeVar("T")


class SuccessfulParseAfterRetries(BaseOutputParser[str]):
    parse_count: int = 0  # Number of times parse has been called
    attemp_count_before_success: (
        int  # Number of times to fail before succeeding  # noqa
    )

    def parse(self, *args: Any, **kwargs: Any) -> str:
        self.parse_count += 1
        if self.parse_count <= self.attemp_count_before_success:
            raise OutputParserException("error")
        return "parsed"


class SuccessfulParseAfterRetriesWithGetFormatInstructions(SuccessfulParseAfterRetries):
    def get_format_instructions(self) -> str:
        return "instructions"


@pytest.mark.parametrize(
    "base_parser",
    [
        SuccessfulParseAfterRetries(attemp_count_before_success=5),
        SuccessfulParseAfterRetriesWithGetFormatInstructions(
            attemp_count_before_success=5
        ),  # noqa: E501
    ],
)
def test_output_fixing_parser_parse(
    base_parser: SuccessfulParseAfterRetries,
) -> None:
    # preparation
    n: int = (
        base_parser.attemp_count_before_success
    )  # Success on the (n+1)-th attempt  # noqa
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
    def fake_llm(prompt: str) -> AIMessage:
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
            attemp_count_before_success=5
        ),  # noqa: E501
    ],
)
async def test_output_fixing_parser_aparse(
    base_parser: SuccessfulParseAfterRetries,
) -> None:
    n: int = (
        base_parser.attemp_count_before_success
    )  # Success on the (n+1)-th attempt   # noqa
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
        parser=base_parser, retry_chain=RunnablePassthrough()
    )
    assert parser.OutputType is base_parser.OutputType


@pytest.mark.parametrize(
    "input,base_parser,retry_chain,expected",
    [
        (
            "2024/07/08",
            DatetimeOutputParser(),
            NAIVE_FIX_PROMPT | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        ),
        (
            # Case: retry_chain.InputType does not have 'instructions' key
            "2024/07/08",
            DatetimeOutputParser(),
            PromptTemplate.from_template("{completion}\n{error}")
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        ),
    ],
)
def test_output_fixing_parser_parse_with_retry_chain(
    input: str,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[Dict[str, Any], str],
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
    assert parser.parse(input) == expected


@pytest.mark.parametrize(
    "input,base_parser,retry_chain,expected",
    [
        (
            "2024/07/08",
            DatetimeOutputParser(),
            NAIVE_FIX_PROMPT | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        ),
        (
            # Case: retry_chain.InputType does not have 'instructions' key
            "2024/07/08",
            DatetimeOutputParser(),
            PromptTemplate.from_template("{completion}\n{error}")
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        ),
    ],
)
async def test_output_fixing_parser_aparse_with_retry_chain(
    input: str,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[Dict[str, Any], str],
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
    assert (await parser.aparse(input)) == expected


def _extract_exception(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Optional[Exception]:
    try:
        func(*args, **kwargs)
    except Exception as e:
        return e
    return None
