from datetime import datetime as dt
from typing import Any, Callable, Dict, Optional, TypeVar

import pytest
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pytest_mock import MockerFixture

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain.output_parsers.retry import (
    NAIVE_RETRY_PROMPT,
    NAIVE_RETRY_WITH_ERROR_PROMPT,
    BaseOutputParser,
    OutputParserException,
    RetryOutputParser,
    RetryWithErrorOutputParser,
)
from langchain.pydantic_v1 import Extra

T = TypeVar("T")


class SuccessfulParseAfterRetries(BaseOutputParser[str]):
    parse_count: int = 0  # Number of times parse has been called
    attemp_count_before_success: int  # Number of times to fail before succeeding
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
    actual = parser.parse_with_prompt("completion", StringPromptValue(text="dummy"))
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
        await parser.aparse_with_prompt("completion", StringPromptValue(text="dummy"))
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
    actual = parser.parse_with_prompt("completion", StringPromptValue(text="dummy"))
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


async def test_retry_with_error_output_parser_aparse_with_prompt_fail() -> None:
    n: int = 5  # Success on the (n+1)-th attempt
    base_parser = SuccessfulParseAfterRetries(attemp_count_before_success=n)
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=RunnablePassthrough(),
        max_retries=n - 1,  # n-1 times to retry, that is, n times call
        legacy=False,
    )
    with pytest.raises(OutputParserException):
        await parser.aparse_with_prompt("completion", StringPromptValue(text="dummy"))
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


@pytest.mark.parametrize(
    "input,prompt,base_parser,retry_chain,expected",
    [
        (
            "2024/07/08",
            StringPromptValue(text="dummy"),
            DatetimeOutputParser(),
            NAIVE_RETRY_PROMPT
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        )
    ],
)
def test_retry_output_parser_parse_with_prompt_with_retry_chain(
    input: str,
    prompt: PromptValue,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[Dict[str, Any], str],
    expected: T,
    mocker: MockerFixture,
) -> None:
    # preparation
    # NOTE: Extra.allow is necessary in order to use spy and mock
    retry_chain.Config.extra = Extra.allow  # type: ignore
    invoke_spy = mocker.spy(retry_chain, "invoke")
    # test
    parser = RetryOutputParser(
        parser=base_parser,
        retry_chain=retry_chain,
        legacy=False,
    )
    assert parser.parse_with_prompt(input, prompt) == expected
    invoke_spy.assert_called_once_with(
        dict(
            prompt=prompt.to_string(),
            completion=input,
        )
    )


@pytest.mark.parametrize(
    "input,prompt,base_parser,retry_chain,expected",
    [
        (
            "2024/07/08",
            StringPromptValue(text="dummy"),
            DatetimeOutputParser(),
            NAIVE_RETRY_PROMPT
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        )
    ],
)
async def test_retry_output_parser_aparse_with_prompt_with_retry_chain(
    input: str,
    prompt: PromptValue,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[Dict[str, Any], str],
    expected: T,
    mocker: MockerFixture,
) -> None:
    # preparation
    # NOTE: Extra.allow is necessary in order to use spy and mock
    retry_chain.Config.extra = Extra.allow  # type: ignore
    ainvoke_spy = mocker.spy(retry_chain, "ainvoke")
    # test
    parser = RetryOutputParser(
        parser=base_parser,
        retry_chain=retry_chain,
        legacy=False,
    )
    assert (await parser.aparse_with_prompt(input, prompt)) == expected
    ainvoke_spy.assert_called_once_with(
        dict(
            prompt=prompt.to_string(),
            completion=input,
        )
    )


@pytest.mark.parametrize(
    "input,prompt,base_parser,retry_chain,expected",
    [
        (
            "2024/07/08",
            StringPromptValue(text="dummy"),
            DatetimeOutputParser(),
            NAIVE_RETRY_WITH_ERROR_PROMPT
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        )
    ],
)
def test_retry_with_error_output_parser_parse_with_prompt_with_retry_chain(
    input: str,
    prompt: PromptValue,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[Dict[str, Any], str],
    expected: T,
    mocker: MockerFixture,
) -> None:
    # preparation
    # NOTE: Extra.allow is necessary in order to use spy and mock
    retry_chain.Config.extra = Extra.allow  # type: ignore
    invoke_spy = mocker.spy(retry_chain, "invoke")
    # test
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=retry_chain,
        legacy=False,
    )
    assert parser.parse_with_prompt(input, prompt) == expected
    invoke_spy.assert_called_once_with(
        dict(
            prompt=prompt.to_string(),
            completion=input,
            error=repr(_extract_exception(base_parser.parse, input)),
        )
    )


@pytest.mark.parametrize(
    "input,prompt,base_parser,retry_chain,expected",
    [
        (
            "2024/07/08",
            StringPromptValue(text="dummy"),
            DatetimeOutputParser(),
            NAIVE_RETRY_WITH_ERROR_PROMPT
            | RunnableLambda(lambda _: "2024-07-08T00:00:00.000000Z"),
            dt(2024, 7, 8),
        )
    ],
)
async def test_retry_with_error_output_parser_aparse_with_prompt_with_retry_chain(
    input: str,
    prompt: PromptValue,
    base_parser: BaseOutputParser[T],
    retry_chain: Runnable[Dict[str, Any], str],
    expected: T,
    mocker: MockerFixture,
) -> None:
    # preparation
    # NOTE: Extra.allow is necessary in order to use spy and mock
    retry_chain.Config.extra = Extra.allow  # type: ignore
    ainvoke_spy = mocker.spy(retry_chain, "ainvoke")
    # test
    parser = RetryWithErrorOutputParser(
        parser=base_parser,
        retry_chain=retry_chain,
        legacy=False,
    )
    assert (await parser.aparse_with_prompt(input, prompt)) == expected
    ainvoke_spy.assert_called_once_with(
        dict(
            prompt=prompt.to_string(),
            completion=input,
            error=repr(_extract_exception(base_parser.parse, input)),
        )
    )


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
