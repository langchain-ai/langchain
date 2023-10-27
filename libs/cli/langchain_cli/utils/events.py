import json
from typing import Any, Dict, List, Optional, TypedDict

import urllib3

WRITE_KEY = "310apTK0HUFl4AOv"


class EventDict(TypedDict):
    event: str
    properties: Optional[Dict[str, Any]]


def create_event(event: EventDict) -> None:
    """
    Creates a new event with the given type and payload.
    """
    data = {
        "write_key": WRITE_KEY,
        "event": event["event"],
        "properties": event.get("properties"),
    }
    try:
        urllib3.request(
            "POST",
            "https://app.firstpartyhq.com/events/v1/track",
            body=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )
    except Exception:
        pass


def create_events(events: List[EventDict]) -> None:
    data = {
        "events": [
            {
                "write_key": WRITE_KEY,
                "event": event["event"],
                "properties": event.get("properties"),
            }
            for event in events
        ]
    }
    try:
        urllib3.request(
            "POST",
            "https://app.firstpartyhq.com/events/v1/track/bulk",
            body=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )
    except Exception:
        pass
