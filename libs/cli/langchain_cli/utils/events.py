import json
from typing import Any, Dict, List, Optional, TypedDict
from urllib import request

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
    data_b = json.dumps(data).encode("utf-8")
    try:
        req = request.Request(
            "https://app.firstpartyhq.com/events/v1/track",
            data=data_b,
            headers={"Content-Type": "application/json"},
        )
        request.urlopen(req)
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
    data_b = json.dumps(data).encode("utf-8")
    try:
        req = request.Request(
            "https://app.firstpartyhq.com/events/v1/track/bulk",
            data=data_b,
            headers={"Content-Type": "application/json"},
        )
        request.urlopen(req)
    except Exception:
        pass
