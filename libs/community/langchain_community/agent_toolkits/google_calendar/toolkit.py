from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_community.tools.google_calendar.create_event import CalendarCreateEvent
from langchain_community.tools.google_calendar.delete_event import CalendarDeleteEvent
from langchain_community.tools.google_calendar.get_calendars_info import (
    GetCalendarsInfo,
)
from langchain_community.tools.google_calendar.move_event import CalendarMoveEvent
from langchain_community.tools.google_calendar.search_events import CalendarSearchEvents
from langchain_community.tools.google_calendar.update_event import CalendarUpdateEvent
from langchain_community.tools.google_calendar.utils import build_resource_service

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource  # type: ignore[import]
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarToolkit(BaseToolkit):
    """Toolkit for interacting with GoogleCalendar.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by reading, creating, updating, deleting
        data associated with this service.

        For example, this toolkit can be used to create events on behalf of the
        associated account.

        See https://python.langchain.com/docs/security for more information.
    """

    api_resource: Resource = Field(default_factory=build_resource_service)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            CalendarCreateEvent(api_resource=self.api_resource),
            CalendarSearchEvents(api_resource=self.api_resource),
            CalendarUpdateEvent(api_resource=self.api_resource),
            GetCalendarsInfo(api_resource=self.api_resource),
            CalendarMoveEvent(api_resource=self.api_resource),
            CalendarDeleteEvent(api_resource=self.api_resource),
        ]
