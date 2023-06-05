import json
import importlib
from typing import Any

from langchain.load.serializable import Serializable


class Reviver:
    secrets_map: dict[str, str]

    def __init__(self, secrets_map: dict[str, str] = None):
        self.secrets_map = secrets_map or dict()

    def __call__(self, value: dict[str, Any]):
        if (
            value.get("lc", None) == 1
            and value.get("type", None) == "secret"
            and value.get("id", None) is not None
        ):
            [key] = value["id"]
            if key in self.secrets_map:
                return self.secrets_map[key]
            else:
                raise KeyError(f'Missing key "{key}" in load(secrets_map)')

        if (
            value.get("lc", None) == 1
            and value.get("type", None) == "not_implemented"
            and value.get("id", None) is not None
        ):
            raise NotImplementedError(
                f"Trying to load an object that doesn't implement serialization: {value}"
            )

        if (
            value.get("lc", None) == 1
            and value.get("type", None) == "constructor"
            and value.get("id", None) is not None
        ):
            [*namespace, name] = value["id"]

            # Currently, we only support langchain imports.
            if namespace[0] != "langchain":
                raise ValueError(f"Invalid namespace: {value}")

            # The root namespace "langchain" is not a valid identifier.
            if len(namespace) == 1:
                raise ValueError(f"Invalid namespace: {value}")

            mod = importlib.import_module(".".join(namespace))
            cls = getattr(mod, name)

            # The class must be a subclass of Serializable.
            if not issubclass(cls, Serializable):
                raise ValueError(f"Invalid namespace: {value}")

            # We don't need to recurse on kwargs
            # as json.loads will do that for us.
            kwargs = value.get("kwargs", dict())
            return cls(**kwargs)

        return value


def loads(text: str, *, secrets_map: dict[str, str] = None):
    return json.loads(text, object_hook=Reviver(secrets_map))
