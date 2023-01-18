"""Chain that calls Google Calendar."""

import datetime
import json
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.utilities.google_calendar.prompts import (
    CLASSIFICATION_PROMPT,
    CREATE_EVENT_PROMPT,
)


class GoogleCalendarAPIWrapper(BaseModel):
    """Wrapper around Google Calendar API.

    To use, you need to create a Google Cloud project and
    enable the Google Calendar API.

    Follow instructions here:
    - https://developers.google.com/calendar/api/quickstart/python

    For pip libraries:
    - pip install --upgrade
    google-api-python-client google-auth-httplib2 google-auth-oauthlib

    OAuth2.0 done through credentials.json folder in the root directory.
    """

    service: Any  #: :meta private:
    google_http_error: Any  #: :meta private:
    creds: Any  #: :meta private:

    SCOPES: List[str] = [
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def create_event(
        self,
        event_summary: str,
        event_start_time: str,
        event_end_time: str,
        user_timezone: str,
        event_location: str = "",
        event_description: str = "",
        # TODO: Implement later
        # event_recurrence:str=None,
        # event_attendees: List[str]=[],
        # event_reminders:str=None,
    ) -> Any:
        """Create an event in the user's calendar."""
        event = {
            "summary": event_summary,
            "location": event_location,
            "description": event_description,
            "start": {
                "dateTime": event_start_time,
                "timeZone": user_timezone,
                # utc
            },
            "end": {
                "dateTime": event_end_time,
                "timeZone": user_timezone,
            },
            # TODO: Implement later
            # "recurrence": [event_recurrence],
            # "attendees": event_attendees,
            # "reminders": event_reminders,
        }
        try:
            created_event = (
                self.service.events().insert(calendarId="primary", body=event).execute()
            )
            return created_event

        except self.google_http_error as error:
            return f"An error occurred: {error}"

    # Not implemented yet
    def view_events(self) -> Any:
        """View all events in the user's calendar."""
        try:
            import datetime

            now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
            events_result = (
                self.service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=10,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])
            if not events:
                print("No upcoming events found.")
                return
            # for event in events:
            #     start = event['start'].get('dateTime', event['start'].get('date'))
            #     print(start, event['summary'])
            return events
        except self.google_http_error as error:
            print(f"An error occurred: {error}")

    # Not implemented yet
    def view_event(self, event_id: str) -> Any:
        """View an event in the user's calendar."""
        try:
            event = (
                self.service.events()
                .get(calendarId="primary", eventId=event_id)
                .execute()
            )
            print(f'Event summary: {event["summary"]}')
            print(f'Event location: {event["location"]}')
            print(f'Event description: {event["description"]}')
            return event
        except self.google_http_error as error:
            print(f"An error occurred: {error}")

    # Not implemented yet
    def reschedule_event(
        self, event_id: str, new_start_time: str, new_end_time: str
    ) -> Any:
        """Reschedule an event in the user's calendar."""
        try:
            event = (
                self.service.events()
                .get(calendarId="primary", eventId=event_id)
                .execute()
            )
            event["start"]["dateTime"] = new_start_time
            event["end"]["dateTime"] = new_end_time
            updated_event = (
                self.service.events()
                .update(calendarId="primary", eventId=event_id, body=event)
                .execute()
            )
            print(f'Event rescheduled: {updated_event.get("htmlLink")}')
            return updated_event
        except self.google_http_error as error:
            print(f"An error occurred: {error}")

    # Not implemented yet
    def delete_event(self, event_id: str) -> Any:
        """Delete an event in the user's calendar."""
        try:
            self.service.events().delete(
                calendarId="primary", eventId=event_id
            ).execute()
            print(f"Event with ID {event_id} has been deleted.")
            return f"Event with ID {event_id} has been deleted."
        except self.google_http_error as error:
            print(f"An error occurred: {error}")

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        #
        # Auth done through OAuth2.0

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError

            SCOPES = [
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/calendar.events",
            ]
            creds: Any = None
            if os.path.exists("token.json"):
                creds = Credentials.from_authorized_user_file("token.json", SCOPES)
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        "credentials.json", SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
            values["service"] = build("calendar", "v3", credentials=creds)
            values["google_http_error"] = HttpError
            values["creds"] = creds

        except ImportError:
            raise ValueError(
                "Could not import google python packages. "
                """Please it install it with `pip install --upgrade
                google-api-python-client google-auth-httplib2 google-auth-oauthlib`."""
            )
        return values

    def run_classification(self, query: str) -> str:
        """Run classification on query."""
        from langchain import LLMChain, OpenAI, PromptTemplate

        prompt = PromptTemplate(
            template=CLASSIFICATION_PROMPT, input_variables=["query"]
        )
        llm_chain = LLMChain(
            llm=OpenAI(temperature=0, model="text-davinci-003"),
            prompt=prompt,
            verbose=True,
        )
        return llm_chain.run(query=query).strip().lower()

    def run_create_event(self, query: str) -> str:
        """Run create event on query."""
        from langchain import LLMChain, OpenAI, PromptTemplate

        # Use a classification chain to classify the query
        date_prompt = PromptTemplate(
            input_variables=["date", "query", "u_timezone"],
            template=CREATE_EVENT_PROMPT,
        )
        create_event_chain = LLMChain(
            llm=OpenAI(temperature=0, model="text-davinci-003"),
            prompt=date_prompt,
            verbose=True,
        )
        date = datetime.datetime.utcnow().isoformat() + "Z"
        u_timezone = str(
            datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        )
        # llm_chain.run(query=query).strip() ouputs a json string
        output = create_event_chain.run(
            query=query, date=date, u_timezone=u_timezone
        ).strip()
        loaded = json.loads(output)
        (
            event_summary,
            event_start_time,
            event_end_time,
            event_location,
            event_description,
            user_timezone,
        ) = loaded.values()

        event = self.create_event(
            event_summary=event_summary,
            event_start_time=event_start_time,
            event_end_time=event_end_time,
            user_timezone=user_timezone,
            event_location=event_location,
            event_description=event_description,
        )
        return "Event created successfully, details: event " + event.get("htmlLink")

    def run(self, query: str) -> str:
        """Ask a question to the notion database."""
        # Use a classification chain to classify the query
        classification = self.run_classification(query)

        if classification == "create_event":
            return self.run_create_event(query)

        # TODO: reschedule_event, view_event, view_events, delete_event

        return "Currently only create event is supported"
