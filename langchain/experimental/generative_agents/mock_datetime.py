import contextlib
import datetime


@contextlib.contextmanager
def mock_now(dt_value):  # type: ignore
    """Context manager for mocking out datetime.now() in unit tests.
    Example:
    with mock_now(datetime.datetime(2011, 2, 3, 10, 11)):
        assert datetime.datetime.now() == datetime.datetime(2011, 2, 3, 10, 11)
    """

    class MockDateTime(datetime.datetime):
        @classmethod
        def now(cls):  # type: ignore
            # Create a copy of dt_value.
            return datetime.datetime(
                dt_value.year,
                dt_value.month,
                dt_value.day,
                dt_value.hour,
                dt_value.minute,
                dt_value.second,
                dt_value.microsecond,
                dt_value.tzinfo,
            )

    real_datetime = datetime.datetime
    datetime.datetime = MockDateTime
    try:
        yield datetime.datetime
    finally:
        datetime.datetime = real_datetime
