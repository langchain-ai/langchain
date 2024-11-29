"""Delete an event in Google Calendar."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.google_calendar.base import GoogleCalendarBaseTool


class DeleteEventSchema(BaseModel):
    """Input for CalendarDeleteEvent."""

    event_id: str = Field(..., description="The event ID to delete.")
    calendar_id: Optional[str] = Field(
        default="primary", description="The origin calendar ID."
    )
    send_updates: Optional[str] = Field(
        default=None,
        description=(
            "Whether to send updates to attendees."
            "Allowed values are 'all', 'externalOnly', or 'none'."
        ),
    )


class CalendarDeleteEvent(GoogleCalendarBaseTool):  # type: ignore[override, override]
    """Tool that delete an event in Google Calendar."""

    name: str = "delete_calendar_event"
    description: str = "Use this tool to delete an event."
    args_schema: Type[DeleteEventSchema] = DeleteEventSchema

    def _run(
        self,
        event_id: str,
        calendar_id: Optional[str] = "primary",
        send_updates: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> None:
        """Run the tool to delete an event in Google Calendar."""
        try:
            self.api_resource.events().delete(
                eventId=event_id, calendarId=calendar_id, sendUpdates=send_updates
            ).execute()
        except Exception as error:
            raise Exception(f"An error occurred: {error}") from error
