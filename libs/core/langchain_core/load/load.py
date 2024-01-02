import importlib
import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.load.mapping import (
    OLD_PROMPT_TEMPLATE_FORMATS,
    SERIALIZABLE_MAPPING,
)
from langchain_core.load.serializable import Serializable

DEFAULT_NAMESPACES = ["langchain", "langchain_core", "langchain_community"]

ALL_SERIALIZABLE_MAPPINGS = {**SERIALIZABLE_MAPPING, **OLD_PROMPT_TEMPLATE_FORMATS}


class Reviver:
    """Reviver for JSON objects."""

    def __init__(
        self,
        secrets_map: Optional[Dict[str, str]] = None,
        valid_namespaces: Optional[List[str]] = None,
    ) -> None:
        self.secrets_map = secrets_map or dict()
        # By default only support langchain, but user can pass in additional namespaces
        self.valid_namespaces = (
            [*DEFAULT_NAMESPACES, *valid_namespaces]
            if valid_namespaces
            else DEFAULT_NAMESPACES
        )

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

            if namespace[0] not in self.valid_namespaces:
                raise ValueError(f"Invalid namespace: {value}")

            # The root namespace "langchain" is not a valid identifier.
            if len(namespace) == 1 and namespace[0] == "langchain":
                raise ValueError(f"Invalid namespace: {value}")

            # If namespace is in known namespaces, try to use mapping
            if namespace[0] in DEFAULT_NAMESPACES:
                # Get the importable path
                key = tuple(namespace + [name])
                if key not in ALL_SERIALIZABLE_MAPPINGS:
                    raise ValueError(
                        "Trying to deserialize something that cannot "
                        "be deserialized in current version of langchain-core: "
                        f"{key}"
                    )
                import_path = ALL_SERIALIZABLE_MAPPINGS[key]
                # Split into module and name
                import_dir, import_obj = import_path[:-1], import_path[-1]
                # Import module
                mod = importlib.import_module(".".join(import_dir))
                # Import class
                cls = getattr(mod, import_obj)
            # Otherwise, load by path
            else:
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


def loads(
    text: str,
    *,
    secrets_map: Optional[Dict[str, str]] = None,
    valid_namespaces: Optional[List[str]] = None,
) -> Any:
    """Revive a LangChain class from a JSON string.
    Equivalent to `load(json.loads(text))`.

    Args:
        text: The string to load.
        secrets_map: A map of secrets to load.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized.

    Returns:
        Revived LangChain objects.
    """
    return json.loads(text, object_hook=Reviver(secrets_map, valid_namespaces))


def load(
    obj: Any,
    *,
    secrets_map: Optional[Dict[str, str]] = None,
    valid_namespaces: Optional[List[str]] = None,
) -> Any:
    """Revive a LangChain class from a JSON object. Use this if you already
    have a parsed JSON object, eg. from `json.load` or `orjson.loads`.

    Args:
        obj: The object to load.
        secrets_map: A map of secrets to load.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized.

    Returns:
        Revived LangChain objects.
    """
    reviver = Reviver(secrets_map, valid_namespaces)

    def _load(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Need to revive leaf nodes before reviving this node
            loaded_obj = {k: _load(v) for k, v in obj.items()}
            return reviver(loaded_obj)
        if isinstance(obj, list):
            return [_load(o) for o in obj]
        return obj

    return _load(obj)
