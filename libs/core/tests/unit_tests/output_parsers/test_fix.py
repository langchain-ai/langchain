"""Tests for OutputFixingParser."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import OutputFixingParser
from langchain_core.output_parsers.base import BaseOutputParser


class MockParser(BaseOutputParser[dict[str, Any]]):
    """A mock parser that fails a configurable number of times before succeeding."""

    fail_count: int = 0
    """Number of times to fail before succeeding."""
    current_failures: int = 0
    """Current number of failures."""
    expected_output: ClassVar[dict[str, Any]] = {"result": "success"}
    """Output to return on success."""

    def parse(self, text: str) -> dict[str, Any]:  # noqa: ARG002
        if self.current_failures < self.fail_count:
            self.current_failures += 1
            msg = f"Parse failed (attempt {self.current_failures})"
            raise OutputParserException(msg)
        return self.expected_output

    async def aparse(self, text: str) -> dict[str, Any]:
        return self.parse(text)

    def get_format_instructions(self) -> str:
        return "Return valid JSON."

    @property
    def _type(self) -> str:
        return "mock"


class MockParserNoInstructions(BaseOutputParser[str]):
    """A mock parser without get_format_instructions."""

    def parse(self, text: str) -> str:  # noqa: ARG002
        msg = "Always fails"
        raise OutputParserException(msg)

    async def aparse(self, text: str) -> str:
        return self.parse(text)

    def get_format_instructions(self) -> str:
        msg = "No format instructions"
        raise NotImplementedError(msg)

    @property
    def _type(self) -> str:
        return "mock_no_instructions"


def test_output_fixing_parser_success_first_try() -> None:
    """Test that parsing succeeds on first try when output is valid."""
    mock_parser = MockParser(fail_count=0)
    mock_llm = MagicMock()

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser.from_llm(
        llm=mock_llm,
        parser=mock_parser,
        max_retries=3,
    )

    result = fixing_parser.parse('{"result": "success"}')

    assert result == {"result": "success"}
    mock_llm.invoke.assert_not_called()


def test_output_fixing_parser_success_after_retry() -> None:
    """Test that parsing succeeds after retrying."""
    mock_parser = MockParser(fail_count=1)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="fixed output")

    # Create a mock chain that returns the fixed output
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "fixed output"

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=3,
    )

    result = fixing_parser.parse("invalid json")

    assert result == {"result": "success"}
    mock_chain.invoke.assert_called_once()


def test_output_fixing_parser_fails_after_max_retries() -> None:
    """Test that parsing fails after exhausting all retries."""
    mock_parser = MockParser(fail_count=10)  # Always fails
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "still invalid"

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=2,
    )

    with pytest.raises(OutputParserException):
        fixing_parser.parse("invalid json")

    # Should have retried max_retries times
    assert mock_chain.invoke.call_count == 2


def test_output_fixing_parser_without_format_instructions() -> None:
    """Test that parsing works when parser doesn't have format instructions."""
    mock_parser = MockParserNoInstructions()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "fixed output"

    fixing_parser: OutputFixingParser[str] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=1,
    )

    with pytest.raises(OutputParserException):
        fixing_parser.parse("invalid")

    # Should have called without instructions
    call_args = mock_chain.invoke.call_args[0][0]
    assert "completion" in call_args
    assert "error" in call_args
    assert "instructions" not in call_args


def test_output_fixing_parser_get_format_instructions() -> None:
    """Test that format instructions are passed through from wrapped parser."""
    mock_parser = MockParser()
    mock_chain = MagicMock()

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=1,
    )

    assert fixing_parser.get_format_instructions() == "Return valid JSON."


def test_output_fixing_parser_output_type() -> None:
    """Test that OutputType is passed through from wrapped parser."""
    mock_parser = MockParser()
    mock_chain = MagicMock()

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=1,
    )

    assert fixing_parser.OutputType == mock_parser.OutputType


@pytest.mark.asyncio
async def test_output_fixing_parser_aparse_success() -> None:
    """Test async parsing succeeds on first try."""
    mock_parser = MockParser(fail_count=0)
    mock_chain = MagicMock()

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=1,
    )

    result = await fixing_parser.aparse('{"result": "success"}')

    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_output_fixing_parser_aparse_with_retry() -> None:
    """Test async parsing with retry."""
    mock_parser = MockParser(fail_count=1)
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "fixed output"

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=3,
    )

    result = await fixing_parser.aparse("invalid json")

    assert result == {"result": "success"}
    mock_chain.ainvoke.assert_called_once()


def test_output_fixing_parser_is_serializable() -> None:
    """Test that OutputFixingParser is marked as serializable."""
    assert OutputFixingParser.is_lc_serializable() is True


def test_output_fixing_parser_type() -> None:
    """Test that parser type is correct."""
    mock_parser = MockParser()
    mock_chain = MagicMock()

    fixing_parser: OutputFixingParser[dict[str, Any]] = OutputFixingParser(
        parser=mock_parser,
        retry_chain=mock_chain,
        max_retries=1,
    )

    assert fixing_parser._type == "output_fixing"
