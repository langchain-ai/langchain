from datetime import datetime

import pytest
from langchain_core.exceptions import OutputParserException

from langchain_classic.output_parsers.datetime import DatetimeOutputParser


def test_datetime_output_parser_parse() -> None:
    parser = DatetimeOutputParser()

    # Test valid input
    date = datetime.now()  # noqa: DTZ005
    datestr = date.strftime(parser.format)
    result = parser.parse(datestr)
    assert result == date

    # Test valid input
    parser.format = "%Y-%m-%dT%H:%M:%S"
    datestr = date.strftime(parser.format)
    result = parser.parse(datestr)
    assert result.year == date.year
    assert result.month == date.month
    assert result.day == date.day
    assert result.hour == date.hour
    assert result.minute == date.minute
    assert result.second == date.second

    # Test valid input
    parser.format = "%H:%M:%S"
    datestr = date.strftime(parser.format)
    result = parser.parse(datestr)
    assert result.hour == date.hour
    assert result.minute == date.minute
    assert result.second == date.second

    # Test invalid input
    with pytest.raises(OutputParserException):
        parser.parse("Invalid date string")
