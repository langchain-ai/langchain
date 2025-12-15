"""Load LangChain objects from JSON strings or objects.

!!! warning
    `load` and `loads` are vulnerable to remote code execution. Never use with untrusted
    input.
"""

import importlib
import json
import os
from collections.abc import Iterable
from typing import Any

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

AllowedObject = type[Serializable]


def _compute_allowed_lc_ids(
    allowed_objects: Iterable[AllowedObject],
    import_mappings: dict[tuple[str, ...], tuple[str, ...]],
) -> set[tuple[str, ...]] | None:
    """Return allowed lc_ids; None means allow any Serializable subclass.

    Examples:
        # Allow any Serializable subclass
        _compute_allowed_lc_ids([Serializable], {}) -> None

        # Allow a specific class
        _compute_allowed_lc_ids([MyPrompt], {}) ->
            {tuple(MyPrompt.lc_id())}
            # e.g. {("langchain_core", "prompts", "MyPrompt")}

        # Include legacy ids that map to the same class
        import_mappings = {("old", "Prompt"): tuple(MyPrompt.lc_id())}
        _compute_allowed_lc_ids([MyPrompt], import_mappings) ->
            {tuple(MyPrompt.lc_id()), ("old", "Prompt")}
    """
    allowed_objects_list = list(allowed_objects)
    if any(obj is Serializable for obj in allowed_objects_list):
        return None

    allowed_lc_ids: set[tuple[str, ...]] = set()
    for allowed_obj in allowed_objects_list:
        if not isinstance(allowed_obj, type) or not issubclass(
            allowed_obj, Serializable
        ):
            msg = "allowed_objects must contain Serializable subclasses."
            raise TypeError(msg)

        lc_id = tuple(allowed_obj.lc_id())
        allowed_lc_ids.add(lc_id)
        # Add legacy ids that map to the same class.
        for mapping_key, mapping_value in import_mappings.items():
            if tuple(mapping_value) == lc_id:
                allowed_lc_ids.add(mapping_key)
    return allowed_lc_ids


class Reviver:
    """Reviver for JSON objects."""

    def __init__(
        self,
        allowed_objects: Iterable[AllowedObject],
        secrets_map: dict[str, str] | None = None,
        valid_namespaces: list[str] | None = None,
        secrets_from_env: bool = True,  # noqa: FBT001,FBT002
        additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]]
        | None = None,
        *,
        ignore_unserializable_fields: bool = False,
    ) -> None:
        """Initialize the reviver.

        Args:
            allowed_objects: Allowed LangChain objects to deserialize. Each entry must
                be a Serializable subclass. Include the base Serializable class to
                allow all Serializable subclasses.
            secrets_map: A map of secrets to load.

                If a secret is not found in the map, it will be loaded from the
                environment if `secrets_from_env` is `True`.
            valid_namespaces: A list of additional namespaces (modules)
                to allow to be deserialized.
            secrets_from_env: Whether to load secrets from the environment.
            additional_import_mappings: A dictionary of additional namespace mappings

                You can use this to override default mappings or add new mappings.
            ignore_unserializable_fields: Whether to ignore unserializable fields.
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
        self.allowed_lc_ids = _compute_allowed_lc_ids(
            allowed_objects, self.import_mappings
        )
        self.ignore_unserializable_fields = ignore_unserializable_fields

    def __call__(self, value: dict[str, Any]) -> Any:
        """Revive the value.

        Args:
            value: The value to revive.

        Returns:
            The revived value.

        Raises:
            ValueError: If the namespace is invalid.
            ValueError: If trying to deserialize something that cannot
                be deserialized in the current version of langchain-core.
            NotImplementedError: If the object is not implemented and
                `ignore_unserializable_fields` is False.
        """
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
                self.allowed_lc_ids is not None
                and mapping_key not in self.allowed_lc_ids
            ):
                msg = (
                    "Deserialization of the requested object is not allowed. "
                    f"Update allowed_objects to include {mapping_key!r}."
                )
                raise ValueError(msg)

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
    allowed_objects: Iterable[AllowedObject],
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = True,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
) -> Any:
    """Revive a LangChain class from a JSON string.

    !!! warning
        This function is vulnerable to remote code execution. Never use with untrusted
        input.

    Equivalent to `load(json.loads(text))`.

    Args:
        text: The string to load.
        allowed_objects: Allowed LangChain objects to deserialize. Each entry must be
            a Serializable subclass. Include the base Serializable class to allow all
            Serializable subclasses.
        secrets_map: A map of secrets to load.

            If a secret is not found in the map, it will be loaded from the environment
            if `secrets_from_env` is `True`.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized.
        secrets_from_env: Whether to load secrets from the environment.
        additional_import_mappings: A dictionary of additional namespace mappings

            You can use this to override default mappings or add new mappings.
        ignore_unserializable_fields: Whether to ignore unserializable fields.

    Returns:
        Revived LangChain objects.
    """
    return json.loads(
        text,
        object_hook=Reviver(
            allowed_objects,
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
    allowed_objects: Iterable[AllowedObject],
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = True,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
) -> Any:
    """Revive a LangChain class from a JSON object.

    !!! warning
        This function is vulnerable to remote code execution. Never use with untrusted
        input.

    Use this if you already have a parsed JSON object,
    eg. from `json.load` or `orjson.loads`.

    Args:
        obj: The object to load.
        allowed_objects: Allowed LangChain objects to deserialize. Each entry must be
            a Serializable subclass. Include the base Serializable class to allow all
            Serializable subclasses.
        secrets_map: A map of secrets to load.

            If a secret is not found in the map, it will be loaded from the environment
            if `secrets_from_env` is `True`.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized.
        secrets_from_env: Whether to load secrets from the environment.
        additional_import_mappings: A dictionary of additional namespace mappings

            You can use this to override default mappings or add new mappings.
        ignore_unserializable_fields: Whether to ignore unserializable fields.

    Returns:
        Revived LangChain objects.
    """
    reviver = Reviver(
        allowed_objects,
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
