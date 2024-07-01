from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnablePassthrough

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain.output_parsers.fix import BaseOutputParser, OutputFixingParser


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


class SuccessfulParseAfterRetriesWithGetFormatInstructions(SuccessfulParseAfterRetries):  # noqa
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
    parser = OutputFixingParser(
        parser=base_parser,
        max_retries=n,  # n times to retry, that is, (n+1) times call
        retry_chain=RunnablePassthrough(),
        legacy=False,
    )
    # test
    assert parser.parse("completion") == "parsed"
    assert base_parser.parse_count == n + 1
    # TODO: test whether "instructions" is passed to the retry_chain


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
    parser = OutputFixingParser(
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
    parser = OutputFixingParser(
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
    parser = OutputFixingParser(
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
def test_output_fixing_parser_output_type(base_parser: BaseOutputParser) -> None:  # noqa: E501
    parser = OutputFixingParser(parser=base_parser, retry_chain=RunnablePassthrough())  # noqa: E501
    assert parser.OutputType is base_parser.OutputType
