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


def dumps(obj: Any, *, pretty: bool = False, **kwargs: Any) -> str:
    """Return a json string representation of an object."""
    if "default" in kwargs:
        raise ValueError("`default` should not be passed to dumps")
    if pretty:
        indent = kwargs.pop("indent", 2)
        return json.dumps(obj, default=default, indent=indent, **kwargs)
    else:
        return json.dumps(obj, default=default, **kwargs)


def dumpd(obj: Any) -> Dict[str, Any]:
    """Return a json dict representation of an object."""
    return json.loads(dumps(obj))
