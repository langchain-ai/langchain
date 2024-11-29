"""Move an event between calendars in Google Calendar."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.google_calendar.base import GoogleCalendarBaseTool


class MoveEventSchema(BaseModel):
    """Input for CalendarMoveEvent."""

    event_id: str = Field(..., description="The event ID to move.")
    origin_calenddar_id: str = Field(..., description="The origin calendar ID.")
    destination_calendar_id: str = Field(
        ..., description="The destination calendar ID."
    )
    send_updates: Optional[str] = Field(
        default=None,
        description=(
            "Whether to send updates to attendees."
            "Allowed values are 'all', 'externalOnly', or 'none'."
        ),
    )


class CalendarMoveEvent(GoogleCalendarBaseTool):  # type: ignore[override, override]
    """Tool that move an event between calendars in Google Calendar."""

    name: str = "move_calendar_event"
    description: str = "Use this tool to move an event between calendars."
    args_schema: Type[MoveEventSchema] = MoveEventSchema

    def _run(
        self,
        event_id: str,
        origin_calendar_id: str,
        destination_calendar_id: str,
        send_updates: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool to update an event in Google Calendar."""
        try:
            result = (
                self.api_resource.events()
                .move(
                    eventId=event_id,
                    calendarId=origin_calendar_id,
                    destination=destination_calendar_id,
                    sendUpdates=send_updates,
                )
                .execute()
            )
            return result.get("htmlLink")
        except Exception as error:
            raise Exception(f"An error occurred: {error}") from error
