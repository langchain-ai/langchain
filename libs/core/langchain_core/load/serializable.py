from abc import ABC
from collections import deque
from dataclasses import MISSING, dataclass, fields, replace
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

from typing_extensions import NotRequired

from langchain_core.load.dataclass_ext import set_init


class BaseSerialized(TypedDict):
    """Base class for serialized objects."""

    lc: int
    id: List[str]
    name: NotRequired[str]
    graph: NotRequired[Dict[str, Any]]


class SerializedConstructor(BaseSerialized):
    """Serialized constructor."""

    type: Literal["constructor"]
    kwargs: Dict[str, Any]


class SerializedSecret(BaseSerialized):
    """Serialized secret."""

    type: Literal["secret"]


class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented."""

    type: Literal["not_implemented"]
    repr: Optional[str]


class Serializable(ABC):
    """Serializable base class.

    This class is used to serialize objects to JSON.

    It relies on the following methods and properties:

    - `is_lc_serializable`: Is this class serializable?
        By design even if a class inherits from Serializable, it is not serializable by
        default. This is to prevent accidental serialization of objects that should not
        be serialized.
    - `get_lc_namespace`: Get the namespace of the langchain object.
        During de-serialization this namespace is used to identify
        the correct class to instantiate.
        Please see the `Reviver` class in `langchain_core.load.load` for more details.
        During de-serialization an additional mapping is handle
        classes that have moved or been renamed across package versions.
    - `lc_secrets`: A map of constructor argument names to secret ids.
    - `lc_attributes`: List of additional attribute names that should be included
        as part of the serialized representation..
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        dataclass(kw_only=True)(cls)
        set_init(cls)

    def __default_init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self.__dict__.items())

    def copy(
        self, deep: Optional[bool] = False, update: Optional[Dict[str, Any]] = None
    ) -> "Serializable":
        """Create a copy of the object."""
        if deep:
            copied = {
                k: v.copy() if hasattr(v, "copy") and callable(v.copy) else v
                for k, v in self.__dict__.items()
            }
        else:
            copied = {}
        return replace(self, **{**copied, **(update or {})})

    def dict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Return the object as a dictionary."""

        def convert(v: Any) -> Any:
            if hasattr(v, "dict") and callable(v.dict):
                return v.dict()
            if _sequence_like(v):
                return v.__class__(convert(x) for x in v)
            if isinstance(v, dict):
                return {k: convert(x) for k, x in v.items()}
            return v

        return {
            k: convert(v)
            for k, v in self.__dict__.items()
            if exclude is None or k not in exclude
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return False

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object.

        For example, if the class is `langchain.llms.openai.OpenAI`, then the
        namespace is ["langchain", "llms", "openai"]
        """
        return cls.__module__.split(".")

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"openai_api_key": "OPENAI_API_KEY"}
        """
        return dict()

    @property
    def lc_attributes(self) -> Dict:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        return {}

    @classmethod
    def lc_id(cls) -> List[str]:
        """A unique identifier for this class for serialization purposes.

        The unique identifier is a list of strings that describes the path
        to the object.
        """
        return [*cls.get_lc_namespace(), cls.__name__]

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        if not self.is_lc_serializable():
            return self.to_json_not_implemented()

        secrets = dict()
        # Get latest values for kwargs if there is an attribute with same name
        lc_kwargs = {
            k: getattr(self, k, v) for k, v in self if _is_field_useful(self, k, v)
        }

        # Merge the lc_secrets and lc_attributes from every class in the MRO
        for cls in [None, *self.__class__.mro()]:
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            if cls:
                deprecated_attributes = [
                    "lc_namespace",
                    "lc_serializable",
                ]

                for attr in deprecated_attributes:
                    if hasattr(cls, attr):
                        raise ValueError(
                            f"Class {self.__class__} has a deprecated "
                            f"attribute {attr}. Please use the corresponding "
                            f"classmethod instead."
                        )

            # Get a reference to self bound to each class in the MRO
            this = cast(Serializable, self if cls is None else super(cls, self))

            secrets.update(this.lc_secrets)
            # Now also add the aliases for the secrets
            # This ensures known secret aliases are hidden.
            # Note: this does NOT hide any other extra kwargs
            # that are not present in the fields.
            for key in list(secrets):
                value = secrets[key]
                if key in this.__fields__:
                    secrets[this.__fields__[key].alias] = value
            lc_kwargs.update(this.lc_attributes)

        # include all secrets, even if not specified in kwargs
        # as these secrets may be passed as an environment variable instead
        for key in secrets.keys():
            secret_value = getattr(self, key, None) or lc_kwargs.get(key)
            if secret_value is not None:
                lc_kwargs.update({key: secret_value})

        return {
            "lc": 1,
            "type": "constructor",
            "id": self.lc_id(),
            "kwargs": lc_kwargs
            if not secrets
            else _replace_secrets(lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        return to_json_not_implemented(self)


def _is_field_useful(inst: Serializable, key: str, value: Any) -> bool:
    """Check if a field is useful as a constructor argument.

    Args:
        inst: The instance.
        key: The key.
        value: The value.

    Returns:
        Whether the field is useful.
    """
    field = next((f for f in fields(inst) if f.name == key), None)
    if not field:
        return False
    if field.metadata.get("exclude"):
        return False
    if not field.init:
        return False
    if field.default is not MISSING:
        return value != field.default
    if field.default_factory is not MISSING:
        return value != field.default_factory()
    return True


def _replace_secrets(
    root: Dict[Any, Any], secrets_map: Dict[str, str]
) -> Dict[Any, Any]:
    result = root.copy()
    for path, secret_id in secrets_map.items():
        [*parts, last] = path.split(".")
        current = result
        for part in parts:
            if part not in current:
                break
            current[part] = current[part].copy()
            current = current[part]
        if last in current:
            current[last] = {
                "lc": 1,
                "type": "secret",
                "id": [secret_id],
            }
    return result


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """Serialize a "not implemented" object.

    Args:
        obj: object to serialize

    Returns:
        SerializedNotImplemented
    """
    _id: List[str] = []
    try:
        if hasattr(obj, "__name__"):
            _id = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            _id = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        pass

    result: SerializedNotImplemented = {
        "lc": 1,
        "type": "not_implemented",
        "id": _id,
        "repr": None,
    }
    try:
        result["repr"] = repr(obj)
    except Exception:
        pass
    return result


def _sequence_like(v: Any) -> bool:
    return isinstance(v, (list, tuple, set, frozenset, deque)) and not _is_namedtuple(
        type(v)
    )


def _is_namedtuple(type_: Type[Any]) -> bool:
    """
    Check if a given class is a named tuple.
    It can be either a `typing.NamedTuple` or `collections.namedtuple`
    """

    return _lenient_issubclass(type_, tuple) and hasattr(type_, "_fields")


def _lenient_issubclass(
    cls: Any, class_or_tuple: Union[Type[Any], Tuple[Type[Any], ...], None]
) -> bool:
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
    except TypeError:
        return False
