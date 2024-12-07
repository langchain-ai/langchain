"""Update an event in Google Calendar."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.google_calendar.base import GoogleCalendarBaseTool
from langchain_community.tools.google_calendar.utils import is_all_day_event


class UpdateEventSchema(BaseModel):
    """Input for CalendarUpdateEvent."""

    event_id: str = Field(..., description="The event ID to update.")
    calendar_id: str = Field(
        default="primary", description="The calendar ID to create the event in."
    )
    summary: Optional[str] = Field(default=None, description="The title of the event.")
    start_datetime: Optional[str] = Field(
        default=None,
        description=(
            "The new start datetime for the event in 'YYYY-MM-DD HH:MM:SS' format. "
            "If the event is an all-day event, set the time to 'YYYY-MM-DD' format."
        ),
    )
    end_datetime: Optional[str] = Field(
        default=None,
        description=(
            "The new end datetime for the event in 'YYYY-MM-DD HH:MM:SS' format. "
            "If the event is an all-day event, set the time to 'YYYY-MM-DD' format."
        ),
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
    send_updates: Optional[str] = Field(
        default=None,
        description=(
            "Whether to send updates to attendees. "
            "Allowed values are 'all', 'externalOnly', or 'none'."
        ),
    )


class CalendarUpdateEvent(GoogleCalendarBaseTool):  # type: ignore[override, override]
    """Tool that updates an event in Google Calendar."""

    name: str = "update_calendar_event"
    description: str = "Use this tool to update an event. "
    args_schema: Type[UpdateEventSchema] = UpdateEventSchema

    def _get_event(self, event_id: str, calendar_id: str = "primary") -> Dict[str, Any]:
        """Get the event by ID."""
        event = (
            self.api_resource.events()
            .get(calendarId=calendar_id, eventId=event_id)
            .execute()
        )
        return event

    def _refactor_event(
        self,
        event: Dict[str, Any],
        summary: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
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
        """Refactor the event body."""
        if summary is not None:
            event["summary"] = summary
        try:
            if start_datetime and end_datetime:
                if is_all_day_event(start_datetime, end_datetime):
                    event["start"] = {"date": start_datetime}
                    event["end"] = {"date": end_datetime}
                else:
                    datetime_format = "%Y-%m-%d %H:%M:%S"
                    timezone = timezone or event["start"]["timeZone"]
                    start_dt = datetime.strptime(start_datetime, datetime_format)
                    end_dt = datetime.strptime(end_datetime, datetime_format)
                    event["start"] = {
                        "dateTime": start_dt.astimezone().isoformat(),
                        "timeZone": timezone,
                    }
                    event["end"] = {
                        "dateTime": end_dt.astimezone().isoformat(),
                        "timeZone": timezone,
                    }
        except ValueError as error:
            raise ValueError("The datetime format is incorrect.") from error
        if (recurrence is not None) and (isinstance(recurrence, dict)):
            recurrence_items = [
                f"{k}={v}" for k, v in recurrence.items() if v is not None
            ]
            event.update({"recurrence": ["RRULE:" + ";".join(recurrence_items)]})
        if location is not None:
            event.update({"location": location})
        if description is not None:
            event.update({"description": description})
        if attendees is not None:
            attendees_emails = []
            email_pattern = r"^[^@]+@[^@]+\.[^@]+$"
            for email in attendees:
                if not re.match(email_pattern, email):
                    raise ValueError(f"Invalid email address: {email}")
                attendees_emails.append({"email": email})
            event.update({"attendees": attendees_emails})
        if reminders is not None:
            if reminders is True:
                event.update({"reminders": {"useDefault": True}})
            elif isinstance(reminders, list):
                for reminder in reminders:
                    if "method" not in reminder or "minutes" not in reminder:
                        raise ValueError(
                            "Each reminder must have 'method' and 'minutes' keys."
                        )
                    if reminder["method"] not in ["email", "popup"]:
                        raise ValueError(
                            "The reminder method must be 'email' or 'popup'."
                        )
                event.update(
                    {"reminders": {"useDefault": False, "overrides": reminders}}
                )
            else:
                event.update({"reminders": {"useDefault": False}})
        if conference_data:
            event.update(
                {
                    "conferenceData": {
                        "createRequest": {
                            "requestId": str(uuid4()),
                            "conferenceSolutionKey": {"type": "hangoutsMeet"},
                        }
                    }
                }
            )
        else:
            event.update({"conferenceData": None})
        if color_id is not None:
            event["colorId"] = color_id
        if transparency is not None:
            event.update({"transparency": transparency})
        return event

    def _run(
        self,
        event_id: str,
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
        send_updates: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool to update an event in Google Calendar."""
        try:
            event = self._get_event(event_id, calendar_id)
            body = self._refactor_event(
                event=event,
                summary=summary,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
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
            result = (
                self.api_resource.events()
                .update(
                    calendarId=calendar_id,
                    eventId=event_id,
                    body=body,
                    conferenceDataVersion=conference_version,
                    sendUpdates=send_updates,
                )
                .execute()
            )
            return result.get("htmlLink")
        except Exception as error:
            raise Exception(f"An error occurred: {error}") from error
