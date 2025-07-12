from datetime import datetime

import pytest
from langchain_core.exceptions import OutputParserException

from langchain.output_parsers.datetime import DatetimeOutputParser


def test_datetime_output_parser_parse() -> None:
    parser = DatetimeOutputParser()

    # Test valid input
    date = datetime.now()
    datestr = date.strftime(parser.format)
    result = parser.parse(datestr)
    assert result == date

    # Test valid input
    parser.format = "%Y-%m-%dT%H:%M:%S"
    date = datetime.now()
    datestr = date.strftime(parser.format)
    result = parser.parse(datestr)
    assert (
        result.year == date.year
        and result.month == date.month
        and result.day == date.day
        and result.hour == date.hour
        and result.minute == date.minute
        and result.second == date.second
    )

    # Test valid input
    parser.format = "%H:%M:%S"
    date = datetime.now()
    datestr = date.strftime(parser.format)
    result = parser.parse(datestr)
    assert (
        result.hour == date.hour
        and result.minute == date.minute
        and result.second == date.second
    )

    # Test invalid input
    with pytest.raises(OutputParserException):
        parser.parse("Invalid date string")
