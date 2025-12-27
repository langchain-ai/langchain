"""Load LangChain objects from JSON strings or objects.

## How it works

Each `Serializable` LangChain object has a unique identifier (its "class path"), which
is a list of strings representing the module path and class name. For example:

- `AIMessage` -> `["langchain_core", "messages", "ai", "AIMessage"]`
- `ChatPromptTemplate` -> `["langchain_core", "prompts", "chat", "ChatPromptTemplate"]`

When deserializing, the class path from the JSON `'id'` field is checked against an
allowlist. If the class is not in the allowlist, deserialization raises a `ValueError`.

## Security model

The `allowed_objects` parameter controls which classes can be deserialized:

- **`'core'` (default)**: Allow classes defined in the serialization mappings for
    langchain_core.
- **`'all'`**: Allow classes defined in the serialization mappings. This
    includes core LangChain types (messages, prompts, documents, etc.) and trusted
    partner integrations. See `langchain_core.load.mapping` for the full list.
- **Explicit list of classes**: Only those specific classes are allowed.

For simple data types like messages and documents, the default allowlist is safe to use.
These classes do not perform side effects during initialization.

!!! note "Side effects in allowed classes"

    Deserialization calls `__init__` on allowed classes. If those classes perform side
    effects during initialization (network calls, file operations, etc.), those side
    effects will occur. The allowlist prevents instantiation of classes outside the
    allowlist, but does not sandbox the allowed classes themselves.

Import paths are also validated against trusted namespaces before any module is
imported.

### Injection protection (escape-based)

During serialization, plain dicts that contain an `'lc'` key are escaped by wrapping
them: `{"__lc_escaped__": {...}}`. During deserialization, escaped dicts are unwrapped
and returned as plain dicts, NOT instantiated as LC objects.

This is an allowlist approach: only dicts explicitly produced by
`Serializable.to_json()` (which are NOT escaped) are treated as LC objects;
everything else is user data.

Even if an attacker's payload includes `__lc_escaped__` wrappers, it will be unwrapped
to plain dicts and NOT instantiated as malicious objects.

## Examples

```python
from langchain_core.load import load
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Use default allowlist (classes from mappings) - recommended
obj = load(data)

# Allow only specific classes (most restrictive)
obj = load(
    data,
    allowed_objects=[
        ChatPromptTemplate,
        AIMessage,
        HumanMessage,
    ],
)
```
"""

import importlib
import json
import os
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from langchain_core._api import beta
from langchain_core.load._validation import _is_escaped_dict, _unescape_value
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

# Cache for the default allowed class paths computed from mappings
# Maps mode ("all" or "core") to the cached set of paths
_default_class_paths_cache: dict[str, set[tuple[str, ...]]] = {}


def _get_default_allowed_class_paths(
    allowed_object_mode: Literal["all", "core"],
) -> set[tuple[str, ...]]:
    """Get the default allowed class paths from the serialization mappings.

    This uses the mappings as the source of truth for what classes are allowed
    by default. Both the legacy paths (keys) and current paths (values) are included.

    Args:
        allowed_object_mode: either `'all'` or `'core'`.

    Returns:
        Set of class path tuples that are allowed by default.
    """
    if allowed_object_mode in _default_class_paths_cache:
        return _default_class_paths_cache[allowed_object_mode]

    allowed_paths: set[tuple[str, ...]] = set()
    for key, value in ALL_SERIALIZABLE_MAPPINGS.items():
        if allowed_object_mode == "core" and value[0] != "langchain_core":
            continue
        allowed_paths.add(key)
        allowed_paths.add(value)

    _default_class_paths_cache[allowed_object_mode] = allowed_paths
    return _default_class_paths_cache[allowed_object_mode]


def _block_jinja2_templates(
    class_path: tuple[str, ...],
    kwargs: dict[str, Any],
) -> None:
    """Block jinja2 templates during deserialization for security.

    Jinja2 templates can execute arbitrary code, so they are blocked by default when
    deserializing objects with `template_format='jinja2'`.

    Note:
        We intentionally do NOT check the `class_path` here to keep this simple and
        future-proof. If any new class is added that accepts `template_format='jinja2'`,
        it will be automatically blocked without needing to update this function.

    Args:
        class_path: The class path tuple being deserialized (unused).
        kwargs: The kwargs dict for the class constructor.

    Raises:
        ValueError: If `template_format` is `'jinja2'`.
    """
    _ = class_path  # Unused - see docstring for rationale. Kept to satisfy signature.
    if kwargs.get("template_format") == "jinja2":
        msg = (
            "Jinja2 templates are not allowed during deserialization for security "
            "reasons. Use 'f-string' template format instead, or explicitly allow "
            "jinja2 by providing a custom init_validator."
        )
        raise ValueError(msg)


def default_init_validator(
    class_path: tuple[str, ...],
    kwargs: dict[str, Any],
) -> None:
    """Default init validator that blocks jinja2 templates.

    This is the default validator used by `load()` and `loads()` when no custom
    validator is provided.

    Args:
        class_path: The class path tuple being deserialized.
        kwargs: The kwargs dict for the class constructor.

    Raises:
        ValueError: If template_format is `'jinja2'`.
    """
    _block_jinja2_templates(class_path, kwargs)


AllowedObject = type[Serializable]
"""Type alias for classes that can be included in the `allowed_objects` parameter.

Must be a `Serializable` subclass (the class itself, not an instance).
"""

InitValidator = Callable[[tuple[str, ...], dict[str, Any]], None]
"""Type alias for a callable that validates kwargs during deserialization.

The callable receives:

- `class_path`: A tuple of strings identifying the class being instantiated
    (e.g., `('langchain', 'schema', 'messages', 'AIMessage')`).
- `kwargs`: The kwargs dict that will be passed to the constructor.

The validator should raise an exception if the object should not be deserialized.
"""


def _compute_allowed_class_paths(
    allowed_objects: Iterable[AllowedObject],
    import_mappings: dict[tuple[str, ...], tuple[str, ...]],
) -> set[tuple[str, ...]]:
    """Return allowed class paths from an explicit list of classes.

    A class path is a tuple of strings identifying a serializable class, derived from
    `Serializable.lc_id()`. For example: `('langchain_core', 'messages', 'AIMessage')`.

    Args:
        allowed_objects: Iterable of `Serializable` subclasses to allow.
        import_mappings: Mapping of legacy class paths to current class paths.

    Returns:
        Set of allowed class paths.

    Example:
        ```python
        # Allow a specific class
        _compute_allowed_class_paths([MyPrompt], {}) ->
            {("langchain_core", "prompts", "MyPrompt")}

        # Include legacy paths that map to the same class
        import_mappings = {("old", "Prompt"): ("langchain_core", "prompts", "MyPrompt")}
        _compute_allowed_class_paths([MyPrompt], import_mappings) ->
            {("langchain_core", "prompts", "MyPrompt"), ("old", "Prompt")}
        ```
    """
    allowed_objects_list = list(allowed_objects)

    allowed_class_paths: set[tuple[str, ...]] = set()
    for allowed_obj in allowed_objects_list:
        if not isinstance(allowed_obj, type) or not issubclass(
            allowed_obj, Serializable
        ):
            msg = "allowed_objects must contain Serializable subclasses."
            raise TypeError(msg)

        class_path = tuple(allowed_obj.lc_id())
        allowed_class_paths.add(class_path)
        # Add legacy paths that map to the same class.
        for mapping_key, mapping_value in import_mappings.items():
            if tuple(mapping_value) == class_path:
                allowed_class_paths.add(mapping_key)
    return allowed_class_paths


class Reviver:
    """Reviver for JSON objects.

    Used as the `object_hook` for `json.loads` to reconstruct LangChain objects from
    their serialized JSON representation.

    Only classes in the allowlist can be instantiated.
    """

    def __init__(
        self,
        allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
        secrets_map: dict[str, str] | None = None,
        valid_namespaces: list[str] | None = None,
        secrets_from_env: bool = False,  # noqa: FBT001,FBT002
        additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]]
        | None = None,
        *,
        ignore_unserializable_fields: bool = False,
        init_validator: InitValidator | None = default_init_validator,
    ) -> None:
        """Initialize the reviver.

        Args:
            allowed_objects: Allowlist of classes that can be deserialized.
                - `'core'` (default): Allow classes defined in the serialization
                    mappings for `langchain_core`.
                - `'all'`: Allow classes defined in the serialization mappings.

                    This includes core LangChain types (messages, prompts, documents,
                    etc.) and trusted partner integrations. See
                    `langchain_core.load.mapping` for the full list.
                - Explicit list of classes: Only those specific classes are allowed.
            secrets_map: A map of secrets to load.
                If a secret is not found in the map, it will be loaded from the
                environment if `secrets_from_env` is `True`.
            valid_namespaces: Additional namespaces (modules) to allow during
                deserialization, beyond the default trusted namespaces.
            secrets_from_env: Whether to load secrets from the environment.
            additional_import_mappings: A dictionary of additional namespace mappings.

                You can use this to override default mappings or add new mappings.

                When `allowed_objects` is `None` (using defaults), paths from these
                mappings are also added to the allowed class paths.
            ignore_unserializable_fields: Whether to ignore unserializable fields.
            init_validator: Optional callable to validate kwargs before instantiation.

                If provided, this function is called with `(class_path, kwargs)` where
                `class_path` is the class path tuple and `kwargs` is the kwargs dict.
                The validator should raise an exception if the object should not be
                deserialized, otherwise return `None`.

                Defaults to `default_init_validator` which blocks jinja2 templates.
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
        # Compute allowed class paths:
        # - "all" -> use default paths from mappings (+ additional_import_mappings)
        # - Explicit list -> compute from those classes
        if allowed_objects in ("all", "core"):
            self.allowed_class_paths: set[tuple[str, ...]] | None = (
                _get_default_allowed_class_paths(
                    cast("Literal['all', 'core']", allowed_objects)
                ).copy()
            )
            # Add paths from additional_import_mappings to the defaults
            if self.additional_import_mappings:
                for key, value in self.additional_import_mappings.items():
                    self.allowed_class_paths.add(key)
                    self.allowed_class_paths.add(value)
        else:
            self.allowed_class_paths = _compute_allowed_class_paths(
                cast("Iterable[AllowedObject]", allowed_objects), self.import_mappings
            )
        self.ignore_unserializable_fields = ignore_unserializable_fields
        self.init_validator = init_validator

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
                self.allowed_class_paths is not None
                and mapping_key not in self.allowed_class_paths
            ):
                msg = (
                    f"Deserialization of {mapping_key!r} is not allowed. "
                    "The default (allowed_objects='core') only permits core "
                    "langchain-core classes. To allow trusted partner integrations, "
                    "use allowed_objects='all'. Alternatively, pass an explicit list "
                    "of allowed classes via allowed_objects=[...]. "
                    "See langchain_core.load.mapping for the full allowlist."
                )
                raise ValueError(msg)

            if (
                namespace[0] not in self.valid_namespaces
                # The root namespace ["langchain"] is not a valid identifier.
                or namespace == ["langchain"]
            ):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)
            # Determine explicit import path
            if mapping_key in self.import_mappings:
                import_path = self.import_mappings[mapping_key]
                # Split into module and name
                import_dir, name = import_path[:-1], import_path[-1]
            elif namespace[0] in DISALLOW_LOAD_FROM_PATH:
                msg = (
                    "Trying to deserialize something that cannot "
                    "be deserialized in current version of langchain-core: "
                    f"{mapping_key}."
                )
                raise ValueError(msg)
            else:
                # Otherwise, treat namespace as path.
                import_dir = namespace

            # Validate import path is in trusted namespaces before importing
            if import_dir[0] not in self.valid_namespaces:
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)

            mod = importlib.import_module(".".join(import_dir))

            cls = getattr(mod, name)

            # The class must be a subclass of Serializable.
            if not issubclass(cls, Serializable):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)

            # We don't need to recurse on kwargs
            # as json.loads will do that for us.
            kwargs = value.get("kwargs", {})

            if self.init_validator is not None:
                self.init_validator(mapping_key, kwargs)

            return cls(**kwargs)

        return value


@beta()
def loads(
    text: str,
    *,
    allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any:
    """Revive a LangChain class from a JSON string.

    Equivalent to `load(json.loads(text))`.

    Only classes in the allowlist can be instantiated. The default allowlist includes
    core LangChain types (messages, prompts, documents, etc.). See
    `langchain_core.load.mapping` for the full list.

    !!! warning "Beta feature"

        This is a beta feature. Please be wary of deploying experimental code to
        production unless you've taken appropriate precautions.

    Args:
        text: The string to load.
        allowed_objects: Allowlist of classes that can be deserialized.

            - `'core'` (default): Allow classes defined in the serialization mappings
                for `langchain_core`.
            - `'all'`: Allow classes defined in the serialization mappings.

                This includes core LangChain types (messages, prompts, documents, etc.)
                and trusted partner integrations. See `langchain_core.load.mapping` for
                the full list.

            - Explicit list of classes: Only those specific classes are allowed.
            - `[]`: Disallow all deserialization (will raise on any object).
        secrets_map: A map of secrets to load.

            If a secret is not found in the map, it will be loaded from the environment
            if `secrets_from_env` is `True`.
        valid_namespaces: Additional namespaces (modules) to allow during
            deserialization, beyond the default trusted namespaces.
        secrets_from_env: Whether to load secrets from the environment.
        additional_import_mappings: A dictionary of additional namespace mappings.

            You can use this to override default mappings or add new mappings.

            When `allowed_objects` is `None` (using defaults), paths from these
            mappings are also added to the allowed class paths.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
        init_validator: Optional callable to validate kwargs before instantiation.

            If provided, this function is called with `(class_path, kwargs)` where
            `class_path` is the class path tuple and `kwargs` is the kwargs dict.
            The validator should raise an exception if the object should not be
            deserialized, otherwise return `None`.

            Defaults to `default_init_validator` which blocks jinja2 templates.

    Returns:
        Revived LangChain objects.

    Raises:
        ValueError: If an object's class path is not in the `allowed_objects` allowlist.
    """
    # Parse JSON and delegate to load() for proper escape handling
    raw_obj = json.loads(text)
    return load(
        raw_obj,
        allowed_objects=allowed_objects,
        secrets_map=secrets_map,
        valid_namespaces=valid_namespaces,
        secrets_from_env=secrets_from_env,
        additional_import_mappings=additional_import_mappings,
        ignore_unserializable_fields=ignore_unserializable_fields,
        init_validator=init_validator,
    )


@beta()
def load(
    obj: Any,
    *,
    allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any:
    """Revive a LangChain class from a JSON object.

    Use this if you already have a parsed JSON object, eg. from `json.load` or
    `orjson.loads`.

    Only classes in the allowlist can be instantiated. The default allowlist includes
    core LangChain types (messages, prompts, documents, etc.). See
    `langchain_core.load.mapping` for the full list.

    !!! warning "Beta feature"

        This is a beta feature. Please be wary of deploying experimental code to
        production unless you've taken appropriate precautions.

    Args:
        obj: The object to load.
        allowed_objects: Allowlist of classes that can be deserialized.

            - `'core'` (default): Allow classes defined in the serialization mappings
                for `langchain_core`.
            - `'all'`: Allow classes defined in the serialization mappings.

                This includes core LangChain types (messages, prompts, documents, etc.)
                and trusted partner integrations. See `langchain_core.load.mapping` for
                the full list.

            - Explicit list of classes: Only those specific classes are allowed.
            - `[]`: Disallow all deserialization (will raise on any object).
        secrets_map: A map of secrets to load.

            If a secret is not found in the map, it will be loaded from the environment
            if `secrets_from_env` is `True`.
        valid_namespaces: Additional namespaces (modules) to allow during
            deserialization, beyond the default trusted namespaces.
        secrets_from_env: Whether to load secrets from the environment.
        additional_import_mappings: A dictionary of additional namespace mappings.

            You can use this to override default mappings or add new mappings.

            When `allowed_objects` is `None` (using defaults), paths from these
            mappings are also added to the allowed class paths.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
        init_validator: Optional callable to validate kwargs before instantiation.

            If provided, this function is called with `(class_path, kwargs)` where
            `class_path` is the class path tuple and `kwargs` is the kwargs dict.
            The validator should raise an exception if the object should not be
            deserialized, otherwise return `None`.

            Defaults to `default_init_validator` which blocks jinja2 templates.

    Returns:
        Revived LangChain objects.

    Raises:
        ValueError: If an object's class path is not in the `allowed_objects` allowlist.

    Example:
        ```python
        from langchain_core.load import load, dumpd
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="Hello")
        data = dumpd(msg)

        # Deserialize using default allowlist
        loaded = load(data)

        # Or with explicit allowlist
        loaded = load(data, allowed_objects=[AIMessage])

        # Or extend defaults with additional mappings
        loaded = load(
            data,
            additional_import_mappings={
                ("my_pkg", "MyClass"): ("my_pkg", "module", "MyClass"),
            },
        )
        ```
    """
    reviver = Reviver(
        allowed_objects,
        secrets_map,
        valid_namespaces,
        secrets_from_env,
        additional_import_mappings,
        ignore_unserializable_fields=ignore_unserializable_fields,
        init_validator=init_validator,
    )

    def _load(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Check for escaped dict FIRST (before recursing).
            # Escaped dicts are user data that should NOT be processed as LC objects.
            if _is_escaped_dict(obj):
                return _unescape_value(obj)

            # Not escaped - recurse into children then apply reviver
            loaded_obj = {k: _load(v) for k, v in obj.items()}
            return reviver(loaded_obj)
        if isinstance(obj, list):
            return [_load(o) for o in obj]
        return obj

    return _load(obj)
