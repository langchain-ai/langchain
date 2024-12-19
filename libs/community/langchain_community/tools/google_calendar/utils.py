"""Google Calendar tool utils."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

from langchain_core.utils import guard_import

if TYPE_CHECKING:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import Resource
    from googleapiclient.discovery import build as build_resource

logger = logging.getLogger(__name__)


def import_google() -> Tuple[Request, Credentials]:
    """Import google libraries.

    Returns:
        Tuple[Request, Credentials]: Request and Credentials classes.
    """
    return (
        guard_import(
            module_name="google.auth.transport.requests",
            pip_name="google-auth-httplib2",
        ).Request,
        guard_import(
            module_name="google.oauth2.credentials", pip_name="google-auth-httplib2"
        ).Credentials,
    )


def import_installed_app_flow() -> InstalledAppFlow:
    """Import InstalledAppFlow class.

    Returns:
        InstalledAppFlow: InstalledAppFlow class.
    """
    return guard_import(
        module_name="google_auth_oauthlib.flow", pip_name="google-auth-oauthlib"
    ).InstalledAppFlow


def import_googleapiclient_resource_builder() -> build_resource:
    """Import googleapiclient.discovery.build function.

    Returns:
        build_resource: googleapiclient.discovery.build function.
    """
    return guard_import(
        module_name="googleapiclient.discovery", pip_name="google-api-python-client"
    ).build


DEFAULT_SCOPES = ["https://www.googleapis.com/auth/calendar"]
DEFAULT_CREDS_TOKEN_FILE = "token.json"
DEFAULT_CLIENT_SECRETS_FILE = "credentials.json"


def get_google_calendar_credentials(
    token_file: Optional[str] = None,
    client_secrets_file: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> Credentials:
    """Get credentials."""
    # From https://developers.google.com/calendar/api/quickstart/python
    Request, Credentials = import_google()
    InstalledAppFlow = import_installed_app_flow()
    creds = None
    scopes = scopes or DEFAULT_SCOPES
    token_file = token_file or DEFAULT_CREDS_TOKEN_FILE
    client_secrets_file = client_secrets_file or DEFAULT_CLIENT_SECRETS_FILE
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # type: ignore[call-arg]
        else:
            # https://developers.google.com/calendar/api/quickstart/python#authorize_credentials_for_a_desktop_application # noqa
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_file, "w") as token:
            token.write(creds.to_json())
    return creds


def build_resource_service(
    credentials: Optional[Credentials] = None,
    service_name: str = "calendar",
    service_version: str = "v3",
) -> Resource:
    """Build a Google Calendar service."""
    credentials = credentials or get_google_calendar_credentials()
    builder = import_googleapiclient_resource_builder()
    return builder(service_name, service_version, credentials=credentials)


def is_all_day_event(start_datetime: str, end_datetime: str) -> bool:
    """Check if the event is all day."""
    try:
        datetime.strptime(start_datetime, "%Y-%m-%d")
        datetime.strptime(end_datetime, "%Y-%m-%d")
        return True
    except ValueError:
        return False
