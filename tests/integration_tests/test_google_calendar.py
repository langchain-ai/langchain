"""Integration test for Google Calendar API Wrapper."""
from langchain.utilities.google_calendar import GoogleCalendarAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    calendar = GoogleCalendarAPIWrapper()
    output = calendar.run("Schedule a birthday party for 2pm tomorrow")
    assert "Event created successfully" in output
