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

    

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        #
        # Auth done through OAuth2.0

        try:
            from langchain.utilities.google_calendar.loader import google_credentials_loader
            # save the values from loader to values
            values.update(google_credentials_loader())

        except ImportError:
            raise ValueError(
                "Could not import google python packages. "
                """Please it install it with `pip install --upgrade
                google-api-python-client google-auth-httplib2 google-auth-oauthlib`."""
            )
        return values



    

    def run(self, query: str) -> str:
        """Ask a question to the notion database."""
        # Use a classification chain to classify the query
        from langchain.chains.llm_google_calendar.base import LLMGoogleCalendarChain
        from langchain import OpenAI

        google_calendar_chain = LLMGoogleCalendarChain(
            llm=OpenAI(temperature=0),
            query=query,
        )
        return google_calendar_chain.run(query)

