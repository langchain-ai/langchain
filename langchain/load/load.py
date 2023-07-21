import importlib
import json
import os
from typing import Any, Dict, Optional

from langchain.load.serializable import Serializable


class Reviver:
    """Reviver for JSON objects."""

    def __init__(self, secrets_map: Optional[Dict[str, str]] = None) -> None:
        self.secrets_map = secrets_map or dict()

    def __call__(self, value: Dict[str, Any]) -> Any:
        if (
            value.get("lc", None) == 1
            and value.get("type", None) == "secret"
            and value.get("id", None) is not None
        ):
            [key] = value["id"]
            if key in self.secrets_map:
                return self.secrets_map[key]
            else:
                if key in os.environ and os.environ[key]:
                    return os.environ[key]
                raise KeyError(f'Missing key "{key}" in load(secrets_map)')

        if (
            value.get("lc", None) == 1
            and value.get("type", None) == "not_implemented"
            and value.get("id", None) is not None
        ):
            raise NotImplementedError(
                "Trying to load an object that doesn't implement "
                f"serialization: {value}"
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


def loads(text: str, *, secrets_map: Optional[Dict[str, str]] = None) -> Any:
    """Load a JSON object from a string.

    Args:
        text: The string to load.
        secrets_map: A map of secrets to load.

    Returns:

    """
    return json.loads(text, object_hook=Reviver(secrets_map))
