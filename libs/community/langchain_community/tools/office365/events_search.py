"""Util that Searches calendar events in Office 365.

Free, but setup is required. See link below.
https://learn.microsoft.com/en-us/graph/auth/
"""

from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT, clean_body


class SearchEventsInput(BaseModel):
    """Input for SearchEmails Tool.

    From https://learn.microsoft.com/en-us/graph/search-query-parameter"""

    start_datetime: str = Field(
        description=(
            " The start datetime for the search query in the following format: "
            ' YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time '
            " components, and the time zone offset is specified as ±hh:mm. "
            ' For example: "2023-06-09T10:30:00+03:00" represents June 9th, '
            " 2023, at 10:30 AM in a time zone with a positive offset of 3 "
            " hours from Coordinated Universal Time (UTC)."
        )
    )
    end_datetime: str = Field(
        description=(
            " The end datetime for the search query in the following format: "
            ' YYYY-MM-DDTHH:MM:SS±hh:mm, where "T" separates the date and time '
            " components, and the time zone offset is specified as ±hh:mm. "
            ' For example: "2023-06-09T10:30:00+03:00" represents June 9th, '
            " 2023, at 10:30 AM in a time zone with a positive offset of 3 "
            " hours from Coordinated Universal Time (UTC)."
        )
    )
    max_results: int = Field(
        default=10,
        description="The maximum number of results to return.",
    )
    truncate: bool = Field(
        default=True,
        description=(
            "Whether the event's body is truncated to meet token number limits. Set to "
            "False for searches that will retrieve small events, otherwise, set to "
            "True."
        ),
    )


class O365SearchEvents(O365BaseTool):  # type: ignore[override, override]
    """Search calendar events in Office 365.

    Free, but setup is required
    """

    name: str = "events_search"
    args_schema: Type[BaseModel] = SearchEventsInput
    description: str = (
        " Use this tool to search for the user's calendar events."
        " The input must be the start and end datetimes for the search query."
        " The output is a JSON list of all the events in the user's calendar"
        " between the start and end times. You can assume that the user can "
        " not schedule any meeting over existing meetings, and that the user "
        "is busy during meetings. Any times without events are free for the user. "
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    def _run(
        self,
        start_datetime: str,
        end_datetime: str,
        max_results: int = 10,
        truncate: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        truncate_limit: int = 150,
    ) -> List[Dict[str, Any]]:
        # Get calendar object
        schedule = self.account.schedule()
        calendar = schedule.get_default_calendar()

        # Process the date range parameters
        start_datetime_query = dt.strptime(start_datetime, UTC_FORMAT)
        end_datetime_query = dt.strptime(end_datetime, UTC_FORMAT)

        # Run the query
        q = calendar.new_query("start").greater_equal(start_datetime_query)
        q.chain("and").on_attribute("end").less_equal(end_datetime_query)
        events = calendar.get_events(query=q, include_recurring=True, limit=max_results)

        # Generate output dict
        output_events = []
        for event in events:
            output_event = {}
            output_event["organizer"] = event.organizer

            output_event["subject"] = event.subject

            if truncate:
                output_event["body"] = clean_body(event.body)[:truncate_limit]
            else:
                output_event["body"] = clean_body(event.body)

            # Get the time zone from the search parameters
            time_zone = start_datetime_query.tzinfo
            # Assign the datetimes in the search time zone
            output_event["start_datetime"] = event.start.astimezone(time_zone).strftime(
                UTC_FORMAT
            )
            output_event["end_datetime"] = event.end.astimezone(time_zone).strftime(
                UTC_FORMAT
            )
            output_event["modified_date"] = event.modified.astimezone(
                time_zone
            ).strftime(UTC_FORMAT)

            output_events.append(output_event)

        return output_events
