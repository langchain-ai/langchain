"""Events utilities."""

from __future__ import annotations

import http.client
import json
from typing import Any, TypedDict

import typer

WRITE_KEY = "310apTK0HUFl4AOv"


class EventDict(TypedDict):
    """Event data structure for analytics tracking.

    Attributes:
        event: The name of the event.
        properties: Optional dictionary of event properties.
    """

    event: str
    properties: dict[str, Any] | None


def create_events(events: list[EventDict]) -> dict[str, Any] | None:
    """Create events.

    Args:
        events: A list of event dictionaries.

    Returns:
        The response from the event tracking service, or None if there was an error.
    """
    try:
        data = {
            "events": [
                {
                    "write_key": WRITE_KEY,
                    "name": event["event"],
                    "properties": event.get("properties"),
                }
                for event in events
            ],
        }

        conn = http.client.HTTPSConnection("app.firstpartyhq.com")

        payload = json.dumps(data)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        conn.request("POST", "/events/v1/track/bulk", payload, headers)

        res = conn.getresponse()

        response_data = json.loads(res.read())
        return response_data if isinstance(response_data, dict) else None
    except (http.client.HTTPException, OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Error sending events: {exc}")
        return None
