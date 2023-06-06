import json
from typing import Any

from langchain.load.serializable import Serializable, to_json_not_implemented


def default(obj: Any) -> Any:
    if isinstance(obj, Serializable):
        return obj.to_json()
    else:
        return to_json_not_implemented(obj)


def dumps(obj: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(obj, default=default, indent=2)
    else:
        return json.dumps(obj, default=default)
