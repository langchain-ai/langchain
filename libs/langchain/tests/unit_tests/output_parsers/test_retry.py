from typing import Any

import pytest
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import RunnablePassthrough

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain.output_parsers.retry import (
    BaseOutputParser,
    OutputParserException,
    RetryOutputParser,
    RetryWithErrorOutputParser,
)


class SuccessfulParseAfterRetries(BaseOutputParser[str]):
    parse_count: int = 0  # Number of times parse has been called
    attemp_count_before_success: (
        int  # Number of times to fail before succeeding  # noqa
    )
    error_msg: str = "error"

    def parse(self, *args: Any, **kwargs: Any) -> str:
        self.parse_count += 1
        if self.parse_count <= self.attemp_count_before_success:
            raise OutputParserException(self.error_msg)
        return "parsed"


def test_retry_output_parser_parse_with_prompt() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n,  # n times to retry, that is, (n+1) times call
        legacy=False,
    )
    actual = parser.parse_with_prompt("completion", StringPromptValue(text="dummy"))  # noqa: E501
    assert actual == "parsed"
    assert base_parser.parse_count == n + 1


def test_retry_output_parser_parse_with_prompt_fail() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n - 1,  # n-1 times to retry, that is, n times call
        legacy=False,
    )
    with pytest.raises(OutputParserException):
        parser.parse_with_prompt("completion", StringPromptValue(text="dummy"))
    assert base_parser.parse_count == n


async def test_retry_output_parser_aparse_with_prompt() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n,  # n times to retry, that is, (n+1) times call
        legacy=False,
    )
    actual = await parser.aparse_with_prompt(
        "completion", StringPromptValue(text="dummy")
    )
    assert actual == "parsed"
    assert base_parser.parse_count == n + 1


async def test_retry_output_parser_aparse_with_prompt_fail() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n - 1,  # n-1 times to retry, that is, n times call
        legacy=False,
    )
    with pytest.raises(OutputParserException):
        await parser.aparse_with_prompt("completion", StringPromptValue(text="dummy"))  # noqa: E501
    assert base_parser.parse_count == n


@pytest.mark.parametrize(
    "base_parser",
    [
        BooleanOutputParser(),
        DatetimeOutputParser(),
    ],
)
def test_retry_output_parser_output_type(base_parser: BaseOutputParser) -> None:
    parser = RetryOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    assert parser.OutputType is base_parser.OutputType


def test_retry_output_parser_parse_is_not_implemented() -> None:
    parser = RetryOutputParser(
        parser=BooleanOutputParser(),
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    with pytest.raises(NotImplementedError):
        parser.parse("completion")


def test_retry_with_error_output_parser_parse_with_prompt() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n,  # n times to retry, that is, (n+1) times call
        legacy=False,
    )
    actual = parser.parse_with_prompt("completion", StringPromptValue(text="dummy"))  # noqa: E501
    assert actual == "parsed"
    assert base_parser.parse_count == n + 1


def test_retry_with_error_output_parser_parse_with_prompt_fail() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n - 1,  # n-1 times to retry, that is, n times call
        legacy=False,
    )
    with pytest.raises(OutputParserException):
        parser.parse_with_prompt("completion", StringPromptValue(text="dummy"))
    assert base_parser.parse_count == n


async def test_retry_with_error_output_parser_aparse_with_prompt() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n,  # n times to retry, that is, (n+1) times call
        legacy=False,
    )
    actual = await parser.aparse_with_prompt(
        "completion", StringPromptValue(text="dummy")
    )
    assert actual == "parsed"
    assert base_parser.parse_count == n + 1


async def test_retry_with_error_output_parser_aparse_with_prompt_fail() -> None:  # noqa: E501
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n - 1,  # n-1 times to retry, that is, n times call
        legacy=False,
    )
    with pytest.raises(OutputParserException):
        await parser.aparse_with_prompt("completion", StringPromptValue(text="dummy"))  # noqa: E501
    assert base_parser.parse_count == n


@pytest.mark.parametrize(
    "base_parser",
    [
        BooleanOutputParser(),
        DatetimeOutputParser(),
    ],
)
def test_retry_with_error_output_parser_output_type(
    base_parser: BaseOutputParser,
) -> None:
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    assert parser.OutputType is base_parser.OutputType


def test_retry_with_error_output_parser_parse_is_not_implemented() -> None:
    parser = RetryWithErrorOutputParser(
        parser=BooleanOutputParser(),
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    with pytest.raises(NotImplementedError):
        parser.parse("completion")
