import http.client
import json
from typing import Any, Dict, List, Optional, TypedDict

WRITE_KEY = "310apTK0HUFl4AOv"


class EventDict(TypedDict):
    event: str
    properties: Optional[Dict[str, Any]]


def create_events(events: List[EventDict]) -> Optional[Any]:
    try:
        data = {
            "events": [
                {
                    "write_key": WRITE_KEY,
                    "name": event["event"],
                    "properties": event.get("properties"),
                }
                for event in events
            ]
        }

        conn = http.client.HTTPSConnection("app.firstpartyhq.com")

        payload = json.dumps(data)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        conn.request("POST", "/events/v1/track/bulk", payload, headers)

        res = conn.getresponse()

        return json.loads(res.read())
    except Exception:
        return None
