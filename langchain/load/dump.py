import json
from typing import Any

from langchain.load.serializable import Serializable


class LangChainJSONEncoder(json.JSONEncoder):
    def __call__(self, obj: Any) -> Any:
        if isinstance(obj, Serializable):
            return obj.to_json()

        return super().default(obj)


def dumps(obj: Any, *, pretty: bool = False) -> str:
    default = LangChainJSONEncoder()
    if pretty:
        return json.dumps(obj, default=default, indent=2)
    else:
        return json.dumps(obj, default=default)
