from unittest.mock import MagicMock

from langchain_community.tools.google_calendar.create_event import CalendarCreateEvent


def test_create_simple_event() -> None:
    """Test google calendar create event."""
    mock_api_resource = MagicMock()
    # bypass pydantic validation as google-api-python-client is not a package dependency
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_description_and_location() -> None:
    """Test google calendar create event with location."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "description": "Event description",
        "location": "Sante Fe, Mexico City",
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_attendees() -> None:
    """Test google calendar create event with attendees."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "attendees": ["fake123@email.com", "123fake@email.com"],
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_reminders() -> None:
    """Test google calendar create event with reminders."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "reminders": [
            {"method": "email", "minutes": 10},
            {"method": "popup", "minutes": 30},
        ],
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_recurrence() -> None:
    """Test google calendar create event with recurrence."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "recurrence": {
            "FREQ": "WEEKLY",
            "COUNT": 10,
        },
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_conference_data() -> None:
    """Test google calendar create event with conference data."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "conference_data": True,
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")
