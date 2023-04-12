"""Load the google credentials. Abracted."""

import os
from typing import Any, Dict, List

def google_credentials_loader() -> Dict[str, Any]:
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

        return {
            "service" : build("calendar", "v3", credentials=creds),
            "google_http_error" : HttpError,
            "creds" : creds
        }
    except ImportError:
        raise ValueError(
            "Could not import google python packages. "
            """Please it install it with `pip install --upgrade
            google-api-python-client google-auth-httplib2 google-auth-oauthlib`."""
        )