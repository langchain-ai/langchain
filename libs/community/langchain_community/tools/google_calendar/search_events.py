"""Search an event in Google Calendar."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo  # Python 3.9+

from langchain_community.tools.google_calendar.base import GoogleCalendarBaseTool


class SearchEventsSchema(BaseModel):
    """Input for CalendarSearchEvents."""

    min_datetime: Optional[str] = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description=(
            "The start datetime for the events in 'YYYY-MM-DD HH:MM:SS' format. "
            f"The current year is {datetime.now().year}."
        ),
    )
    max_datetime: Optional[str] = Field(
        ..., description="The end datetime for the events."
    )
    max_results: int = Field(
        default=10, description="The maximum number of results to return."
    )
    single_events: bool = Field(
        default=True,
        description=(
            "Whether to expand recurring events into instances and only return single "
            "one-off events and instances of recurring events."
            "'startTime' or 'updated'."
        ),
    )
    order_by: str = Field(
        default="startTime",
        description="The order of the events, either 'startTime' or 'updated'.",
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "Free text search terms to find events, "
            "that match these terms in the following fields: "
            "summary, description, location, attendee's displayName, attendee's email, "
            "organizer's displayName, organizer's email."
        ),
    )


class CalendarSearchEvents(GoogleCalendarBaseTool):  # type: ignore[override, override]
    """Tool that retrieves events from Google Calendar."""

    name: str = "search_events"
    description: str = "Use this tool to search events in the calendar."
    args_schema: Type[SearchEventsSchema] = SearchEventsSchema

    def _get_calendars_info(self) -> List[Dict[str, Any]]:
        """Get the calendars info."""
        calendars = self.api_resource.calendarList().list().execute()
        return calendars.get("items", [])

    def _get_calendar_timezone(
        self, calendars_info: List[Dict[str, Any]], calendar_id: str
    ) -> Optional[str]:
        """Get the timezone of the current calendar."""
        for cal in calendars_info:
            if cal["id"] == calendar_id:
                return cal.get("timeZone")
        return None

    def _get_calendar_ids(self, calendars_info: List[Dict[str, Any]]) -> List[str]:
        """Get the calendar IDs."""
        return [cal["id"] for cal in calendars_info if cal.get("selected")]

    def _process_data_events(
        self, events_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Optional[str]]]:
        """Process the data events."""
        simplified_data = []
        for data in events_data:
            event_dict = {
                "id": data.get("id"),
                "htmlLink": data.get("htmlLink"),
                "summary": data.get("summary"),
                "creator": data.get("creator", {}).get("email"),
                "organizer": data.get("organizer", {}).get("email"),
                "start": data.get("start", {}).get("dateTime")
                or data.get("start", {}).get("date"),
                "end": data.get("end", {}).get("dateTime")
                or data.get("end", {}).get("date"),
            }
            simplified_data.append(event_dict)
        return simplified_data

    def _run(
        self,
        min_datetime: Optional[str],
        max_datetime: Optional[str],
        max_results: int = 10,
        single_events: bool = True,
        order_by: str = "startTime",
        query: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """Run the tool to search events in Google Calendar."""
        try:
            calendars_info = self._get_calendars_info()
            calendars = self._get_calendar_ids(calendars_info)
            events = []
            for calendar in calendars:
                tz_name = self._get_calendar_timezone(calendars_info, calendar)
                if tz_name:
                    calendar_tz = ZoneInfo(tz_name)
                else:
                    calendar_tz = None
                time_min = None
                if min_datetime:
                    time_min = (
                        datetime.strptime(min_datetime, "%Y-%m-%d %H:%M:%S")
                        .astimezone(calendar_tz)
                        .isoformat()
                    )
                time_max = None
                if max_datetime:
                    time_max = (
                        datetime.strptime(max_datetime, "%Y-%m-%d %H:%M:%S")
                        .astimezone(calendar_tz)
                        .isoformat()
                    )
                events_result = (
                    self.api_resource.events()
                    .list(
                        calendarId=calendar,
                        timeMin=time_min,
                        timeMax=time_max,
                        maxResults=max_results,
                        singleEvents=single_events,
                        orderBy=order_by,
                        q=query,
                    )
                    .execute()
                )
                cal_events = events_result.get("items", [])
                events.extend(cal_events)
            return self._process_data_events(events)
        except Exception as error:
            raise Exception(
                f"An error occurred while fetching events: {error}"
            ) from error
