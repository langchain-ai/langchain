"""Create an event in Google Calendar."""  # NUEVO

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.google_calendar.base import GoogleCalendarBaseTool
from langchain_community.tools.google_calendar.utils import is_all_day_event


def get_current_datetime() -> str:
    """Get the current datetime."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CreateEventSchema(BaseModel):
    """Input for CalendarCreateEvent."""

    summary: str = Field(..., description="The title of the event.")
    start_datetime: str = Field(
        default=get_current_datetime(),
        description=(
            "The start datetime for the event in 'YYYY-MM-DD HH:MM:SS' format."
            f"The current year is {datetime.now().year}."
            "If the event is an all-day event, set the time to 'YYYY-MM-DD' format."
        ),
    )
    end_datetime: str = Field(
        ...,
        description=(
            "The end datetime for the event in 'YYYY-MM-DD HH:MM:SS' format. "
            "If the event is an all-day event, set the time to 'YYYY-MM-DD' format."
        ),
    )
    calendar_id: str = Field(
        default="primary", description="The calendar ID to create the event in."
    )
    timezone: Optional[str] = Field(
        default=None, description="The timezone of the event."
    )
    recurrence: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "The recurrence of the event. "
            "Format: {'FREQ': <'DAILY' or 'WEEKLY'>, 'INTERVAL': <number>, "
            "'COUNT': <number or None>, 'UNTIL': <'YYYYMMDD' or None>, "
            "'BYDAY': <'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU' or None>}. "
            "Use either COUNT or UNTIL, but not both; set the other to None."
        ),
    )
    location: Optional[str] = Field(
        default=None, description="The location of the event."
    )
    description: Optional[str] = Field(
        default=None, description="The description of the event."
    )
    attendees: Optional[List[str]] = Field(
        default=None, description="A list of attendees' email addresses for the event."
    )
    reminders: Union[None, bool, List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Reminders for the event. "
            "Set to True for default reminders, or provide a list like "
            "[{'method': 'email', 'minutes': <minutes>}, ...]. "
            "Valid methods are 'email' and 'popup'."
        ),
    )
    conference_data: Optional[bool] = Field(
        default=None, description="Whether to include conference data."
    )
    color_id: Optional[str] = Field(
        default=None,
        description=(
            "The color ID of the event. None for default. "
            "'1': Lavender, '2': Sage, '3': Grape, '4': Flamingo, '5': Banana, "
            "'6': Tangerine, '7': Peacock, '8': Graphite, '9': Blueberry, "
            "'10': Basil, '11': Tomato."
        ),
    )
    transparency: Optional[str] = Field(
        default=None,
        description=(
            "User availability for the event."
            "transparent for available and opaque for busy."
        ),
    )


class CalendarCreateEvent(GoogleCalendarBaseTool):  # type: ignore[override, override]
    """Tool that creates an event in Google Calendar."""

    name: str = "create_calendar_event"
    description: str = (
        "Use this tool to create an event. "
        "The input must include the summary, start, and end datetime for the event."
    )
    args_schema: Type[CreateEventSchema] = CreateEventSchema

    def _get_timezone(self, calendar_id: str) -> str:
        """Get the timezone of the specified calendar."""
        calendars = self.api_resource.calendarList().list().execute().get("items", [])
        if not calendars:
            raise ValueError("No calendars found.")

        if calendar_id == "primary":
            return calendars[0]["timeZone"]
        else:
            for item in calendars:
                if item["id"] == calendar_id and item["accessRole"] != "reader":
                    return item["timeZone"]
            raise ValueError(f"Timezone not found for calendar ID: {calendar_id}")

    def _prepare_event(
        self,
        summary: str,
        start_datetime: str,
        end_datetime: str,
        calendar_id: str = "primary",
        timezone: Optional[str] = None,
        recurrence: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None,
        description: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        reminders: Union[None, bool, List[Dict[str, Any]]] = None,
        conference_data: Optional[bool] = None,
        color_id: Optional[str] = None,
        transparency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare the event body."""
        timezone = timezone or self._get_timezone(calendar_id)
        try:
            if is_all_day_event(start_datetime, end_datetime):
                start = {"date": start_datetime}
                end = {"date": end_datetime}
            else:
                datetime_format = "%Y-%m-%d %H:%M:%S"
                start_dt = datetime.strptime(start_datetime, datetime_format)
                end_dt = datetime.strptime(end_datetime, datetime_format)
                start = {
                    "dateTime": start_dt.astimezone().isoformat(),
                    "timeZone": timezone,
                }
                end = {
                    "dateTime": end_dt.astimezone().isoformat(),
                    "timeZone": timezone,
                }
        except ValueError as error:
            raise ValueError("The datetime format is incorrect.") from error
        recurrence_data = None
        if recurrence:
            if isinstance(recurrence, dict):
                recurrence_items = [
                    f"{k}={v}" for k, v in recurrence.items() if v is not None
                ]
                recurrence_data = "RRULE:" + ";".join(recurrence_items)
        attendees_emails: List[Dict[str, str]] = []
        if attendees:
            email_pattern = r"^[^@]+@[^@]+\.[^@]+$"
            for email in attendees:
                if not re.match(email_pattern, email):
                    raise ValueError(f"Invalid email address: {email}")
                attendees_emails.append({"email": email})
        reminders_info: Dict[str, Union[bool, List[Dict[str, Any]]]] = {}
        if reminders is True:
            reminders_info.update({"useDefault": True})
        elif isinstance(reminders, list):
            for reminder in reminders:
                if "method" not in reminder or "minutes" not in reminder:
                    raise ValueError(
                        "Each reminder must have 'method' and 'minutes' keys."
                    )
                if reminder["method"] not in ["email", "popup"]:
                    raise ValueError("The reminder method must be 'email' or 'popup")
            reminders_info.update({"useDefault": False, "overrides": reminders})
        else:
            reminders_info.update({"useDefault": False})
        conference_data_info = None
        if conference_data:
            conference_data_info = {
                "createRequest": {
                    "requestId": str(uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        event_body: Dict[str, Any] = {"summary": summary, "start": start, "end": end}
        if location:
            event_body["location"] = location
        if description:
            event_body["description"] = description
        if recurrence_data:
            event_body["recurrence"] = [recurrence_data]
        if len(attendees_emails) > 0:
            event_body["attendees"] = attendees_emails
        if len(reminders_info) > 0:
            event_body["reminders"] = reminders_info
        if conference_data_info:
            event_body["conferenceData"] = conference_data_info
        if color_id:
            event_body["colorId"] = color_id
        if transparency:
            event_body["transparency"] = transparency
        return event_body

    def _run(
        self,
        summary: str,
        start_datetime: str,
        end_datetime: str,
        calendar_id: str = "primary",
        timezone: Optional[str] = None,
        recurrence: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None,
        description: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        reminders: Union[None, bool, List[Dict[str, Any]]] = None,
        conference_data: Optional[bool] = None,
        color_id: Optional[str] = None,
        transparency: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool to create an event in Google Calendar."""
        try:
            body = self._prepare_event(
                summary=summary,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                calendar_id=calendar_id,
                timezone=timezone,
                recurrence=recurrence,
                location=location,
                description=description,
                attendees=attendees,
                reminders=reminders,
                conference_data=conference_data,
                color_id=color_id,
                transparency=transparency,
            )
            conference_version = 1 if conference_data else 0
            event = (
                self.api_resource.events()
                .insert(
                    calendarId=calendar_id,
                    body=body,
                    conferenceDataVersion=conference_version,
                )
                .execute()
            )
            return f"Event created: {event.get('htmlLink')}"
        except Exception as error:
            raise Exception(f"An error occurred: {error}") from error
