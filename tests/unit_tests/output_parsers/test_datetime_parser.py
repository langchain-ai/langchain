from datetime import datetime
from time import sleep

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
    try:
        sleep(0.001)
        datestr = date.strftime(parser.format)
        result = parser.parse(datestr)
        assert result == date
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
