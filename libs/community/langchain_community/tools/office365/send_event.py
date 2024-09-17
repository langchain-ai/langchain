"""Util that sends calendar events in Office 365.

Free, but setup is required. See link below.
https://learn.microsoft.com/en-us/graph/auth/
"""

from datetime import datetime as dt
from typing import List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT


class SendEventSchema(BaseModel):
    """Input for CreateEvent Tool."""

    body: str = Field(
        ...,
        description="The message body to include in the event.",
    )
    attendees: List[str] = Field(
        ...,
        description="The list of attendees for the event.",
    )
    subject: str = Field(
        ...,
        description="The subject of the event.",
    )
    start_datetime: str = Field(
        description=" The start datetime for the event in the following format: "
        ' YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time '
        " components, and the time zone offset is specified as ±hh:mm. "
        ' For example: "2023-06-09T10:30:00+03:00" represents June 9th, '
        " 2023, at 10:30 AM in a time zone with a positive offset of 3 "
        " hours from Coordinated Universal Time (UTC).",
    )
    end_datetime: str = Field(
        description=" The end datetime for the event in the following format: "
        ' YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time '
        " components, and the time zone offset is specified as ±hh:mm. "
        ' For example: "2023-06-09T10:30:00+03:00" represents June 9th, '
        " 2023, at 10:30 AM in a time zone with a positive offset of 3 "
        " hours from Coordinated Universal Time (UTC).",
    )


class O365SendEvent(O365BaseTool):
    """Tool for sending calendar events in Office 365."""

    name: str = "send_event"
    description: str = (
        "Use this tool to create and send an event with the provided event fields."
    )
    args_schema: Type[SendEventSchema] = SendEventSchema

    def _run(
        self,
        body: str,
        attendees: List[str],
        subject: str,
        start_datetime: str,
        end_datetime: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Get calendar object
        schedule = self.account.schedule()
        calendar = schedule.get_default_calendar()

        event = calendar.new_event()

        event.body = body
        event.subject = subject
        event.start = dt.strptime(start_datetime, UTC_FORMAT)
        event.end = dt.strptime(end_datetime, UTC_FORMAT)
        for attendee in attendees:
            event.attendees.add(attendee)

        # TO-DO: Look into PytzUsageWarning
        event.save()

        output = "Event sent: " + str(event)
        return output
