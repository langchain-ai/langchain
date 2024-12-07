"""Get information about the calendars in Google Calendar."""

from typing import Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_community.tools.google_calendar.base import GoogleCalendarBaseTool


class GetCalendarsInfo(GoogleCalendarBaseTool):  # type: ignore[override, override]
    """Tool that get information about the calendars in Google Calendar."""

    name: str = "get_calendars_info"
    description: str = (
        "Use this tool to get information about the calendars in Google Calendar."
    )

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, str]]:
        """Run the tool to get information about the calendars in Google Calendar."""
        try:
            calendars = self.api_resource.calendarList().list().execute()
            data = []
            for item in calendars.get("items", []):
                if item["accessRole"] != "reader":
                    data.append(
                        {
                            "id": item["id"],
                            "summary": item["summary"],
                            "timeZone": item["timeZone"],
                        }
                    )
            return data
        except Exception as error:
            raise Exception(f"An error occurred: {error}") from error
