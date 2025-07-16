"""Load LangChain objects from JSON strings or objects."""

import importlib
import json
import os
from typing import Any, Optional

from langchain_core._api import beta
from langchain_core.load.mapping import (
    _JS_SERIALIZABLE_MAPPING,
    _OG_SERIALIZABLE_MAPPING,
    OLD_CORE_NAMESPACES_MAPPING,
    SERIALIZABLE_MAPPING,
)
from langchain_core.load.serializable import Serializable

DEFAULT_NAMESPACES = [
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_anthropic",
    "langchain_groq",
    "langchain_google_genai",
    "langchain_aws",
    "langchain_openai",
    "langchain_google_vertexai",
    "langchain_mistralai",
    "langchain_fireworks",
    "langchain_xai",
    "langchain_sambanova",
    "langchain_perplexity",
]
# Namespaces for which only deserializing via the SERIALIZABLE_MAPPING is allowed.
# Load by path is not allowed.
DISALLOW_LOAD_FROM_PATH = [
    "langchain_community",
    "langchain",
]

ALL_SERIALIZABLE_MAPPINGS = {
    **SERIALIZABLE_MAPPING,
    **OLD_CORE_NAMESPACES_MAPPING,
    **_OG_SERIALIZABLE_MAPPING,
    **_JS_SERIALIZABLE_MAPPING,
}


class Reviver:
    """Reviver for JSON objects."""

    def __init__(
        self,
        secrets_map: Optional[dict[str, str]] = None,
        valid_namespaces: Optional[list[str]] = None,
        secrets_from_env: bool = True,  # noqa: FBT001,FBT002
        additional_import_mappings: Optional[
            dict[tuple[str, ...], tuple[str, ...]]
        ] = None,
        *,
        ignore_unserializable_fields: bool = False,
    ) -> None:
        """Initialize the reviver.

        Args:
            secrets_map: A map of secrets to load. If a secret is not found in
                the map, it will be loaded from the environment if `secrets_from_env`
                is True. Defaults to None.
            valid_namespaces: A list of additional namespaces (modules)
                to allow to be deserialized. Defaults to None.
            secrets_from_env: Whether to load secrets from the environment.
                Defaults to True.
            additional_import_mappings: A dictionary of additional namespace mappings
                You can use this to override default mappings or add new mappings.
                Defaults to None.
            ignore_unserializable_fields: Whether to ignore unserializable fields.
                Defaults to False.
        """
        self.secrets_from_env = secrets_from_env
        self.secrets_map = secrets_map or {}
        # By default, only support langchain, but user can pass in additional namespaces
        self.valid_namespaces = (
            [*DEFAULT_NAMESPACES, *valid_namespaces]
            if valid_namespaces
            else DEFAULT_NAMESPACES
        )
        self.additional_import_mappings = additional_import_mappings or {}
        self.import_mappings = (
            {
                **ALL_SERIALIZABLE_MAPPINGS,
                **self.additional_import_mappings,
            }
            if self.additional_import_mappings
            else ALL_SERIALIZABLE_MAPPINGS
        )
        self.ignore_unserializable_fields = ignore_unserializable_fields

    def __call__(self, value: dict[str, Any]) -> Any:
        """Revive the value."""
        if (
            value.get("lc") == 1
            and value.get("type") == "secret"
            and value.get("id") is not None
        ):
            [key] = value["id"]
            if key in self.secrets_map:
                return self.secrets_map[key]
            if self.secrets_from_env and key in os.environ and os.environ[key]:
                return os.environ[key]
            return None

        if (
            value.get("lc") == 1
            and value.get("type") == "not_implemented"
            and value.get("id") is not None
        ):
            if self.ignore_unserializable_fields:
                return None
            msg = (
                "Trying to load an object that doesn't implement "
                f"serialization: {value}"
            )
            raise NotImplementedError(msg)

        if (
            value.get("lc") == 1
            and value.get("type") == "constructor"
            and value.get("id") is not None
        ):
            [*namespace, name] = value["id"]
            mapping_key = tuple(value["id"])

            if (
                namespace[0] not in self.valid_namespaces
                # The root namespace ["langchain"] is not a valid identifier.
                or namespace == ["langchain"]
            ):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)
            # Has explicit import path.
            if mapping_key in self.import_mappings:
                import_path = self.import_mappings[mapping_key]
                # Split into module and name
                import_dir, name = import_path[:-1], import_path[-1]
                # Import module
                mod = importlib.import_module(".".join(import_dir))
            elif namespace[0] in DISALLOW_LOAD_FROM_PATH:
                msg = (
                    "Trying to deserialize something that cannot "
                    "be deserialized in current version of langchain-core: "
                    f"{mapping_key}."
                )
                raise ValueError(msg)
            # Otherwise, treat namespace as path.
            else:
                mod = importlib.import_module(".".join(namespace))

            cls = getattr(mod, name)

            # The class must be a subclass of Serializable.
            if not issubclass(cls, Serializable):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)

            # We don't need to recurse on kwargs
            # as json.loads will do that for us.
            kwargs = value.get("kwargs", {})
            return cls(**kwargs)

        return value


@beta()
def loads(
    text: str,
    *,
    secrets_map: Optional[dict[str, str]] = None,
    valid_namespaces: Optional[list[str]] = None,
    secrets_from_env: bool = True,
    additional_import_mappings: Optional[dict[tuple[str, ...], tuple[str, ...]]] = None,
    ignore_unserializable_fields: bool = False,
) -> Any:
    """Revive a LangChain class from a JSON string.

    Equivalent to `load(json.loads(text))`.

    Args:
        text: The string to load.
        secrets_map: A map of secrets to load. If a secret is not found in
            the map, it will be loaded from the environment if `secrets_from_env`
            is True. Defaults to None.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized. Defaults to None.
        secrets_from_env: Whether to load secrets from the environment.
            Defaults to True.
        additional_import_mappings: A dictionary of additional namespace mappings
            You can use this to override default mappings or add new mappings.
            Defaults to None.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
            Defaults to False.

    Returns:
        Revived LangChain objects.
    """
    return json.loads(
        text,
        object_hook=Reviver(
            secrets_map,
            valid_namespaces,
            secrets_from_env,
            additional_import_mappings,
            ignore_unserializable_fields=ignore_unserializable_fields,
        ),
    )


@beta()
def load(
    obj: Any,
    *,
    secrets_map: Optional[dict[str, str]] = None,
    valid_namespaces: Optional[list[str]] = None,
    secrets_from_env: bool = True,
    additional_import_mappings: Optional[dict[tuple[str, ...], tuple[str, ...]]] = None,
    ignore_unserializable_fields: bool = False,
) -> Any:
    """Revive a LangChain class from a JSON object.

    Use this if you already have a parsed JSON object,
    eg. from `json.load` or `orjson.loads`.

    Args:
        obj: The object to load.
        secrets_map: A map of secrets to load. If a secret is not found in
            the map, it will be loaded from the environment if `secrets_from_env`
            is True. Defaults to None.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized. Defaults to None.
        secrets_from_env: Whether to load secrets from the environment.
            Defaults to True.
        additional_import_mappings: A dictionary of additional namespace mappings
            You can use this to override default mappings or add new mappings.
            Defaults to None.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
            Defaults to False.

    Returns:
        Revived LangChain objects.
    """
    reviver = Reviver(
        secrets_map,
        valid_namespaces,
        secrets_from_env,
        additional_import_mappings,
        ignore_unserializable_fields=ignore_unserializable_fields,
    )

    def _load(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Need to revive leaf nodes before reviving this node
            loaded_obj = {k: _load(v) for k, v in obj.items()}
            return reviver(loaded_obj)
        if isinstance(obj, list):
            return [_load(o) for o in obj]
        return obj

    return _load(obj)
