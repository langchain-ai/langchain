"""Integration test for Wolfram Alpha API Wrapper."""
from langchain.utilities.google_calendar import GoogleCalendarAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = GoogleCalendarAPIWrapper()
    output = search.run("Schedule a birthday party for 2pm tomorrow")
    assert "Event created successfully" in output
