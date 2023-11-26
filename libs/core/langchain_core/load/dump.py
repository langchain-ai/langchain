import json
from typing import Any, Dict

from langchain_core.load.serializable import Serializable, to_json_not_implemented


def default(obj: Any) -> Any:
    """Return a default value for a Serializable object or
    a SerializedNotImplemented object."""
    if isinstance(obj, Serializable):
        return obj.to_json()
    else:
        return to_json_not_implemented(obj)


def dumps(obj: Any, *, pretty: bool = False) -> str:
    """Return a json string representation of an object."""
    if pretty:
        return json.dumps(obj, default=default, indent=2)
    else:
        return json.dumps(obj, default=default)


def dumpd(obj: Any) -> Dict[str, Any]:
    """Return a json dict representation of an object."""
    return json.loads(dumps(obj))
