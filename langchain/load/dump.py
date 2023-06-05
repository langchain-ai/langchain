import json
from typing import Any

from langchain.load.serializable import Serializable


def default(obj: Any) -> Any:
    if isinstance(obj, Serializable):
        return obj.to_json()

    return json.JSONEncoder.default(json.JSONEncoder, obj)


def dumps(obj: any, *, pretty=False):
    if pretty:
        kwargs = {"indent": 2}
    else:
        kwargs = {}
    return json.dumps(obj, default=default, **kwargs)
