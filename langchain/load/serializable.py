from abc import ABC
from typing import Any, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field


class BaseSerialized(TypedDict):
    lc: int
    id: List[str]


class SerializedConstructor(BaseSerialized):
    type: Literal["constructor"]
    kwargs: Dict[str, Any]


class SerializedSecret(BaseSerialized):
    type: Literal["secret"]


class SerializedNotImplemented(BaseSerialized):
    type: Literal["not_implemented"]


class Serializable(BaseModel, ABC):
    @property
    def lc_namespace(self) -> List[str]:
        """
        Return the namespace of the langchain object.
        eg. ["langchain", "llms", "openai"]
        """
        return self.__class__.__module__.split(".")

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """
        Return a map of constructor argument names to secret ids.
        eg. {"openai_api_key": "OPENAI_API_KEY"}
        """
        return dict()

    @property
    def lc_attributes(self) -> List[str]:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return []

    lc_kwargs: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lc_kwargs = kwargs

    def to_json(self) -> SerializedConstructor:
        secrets = dict()

        # Merge the lc_secrets from every class in the MRO
        for cls in self.__class__.mro():
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            # mypy doesn't understand this, but it is correct
            secrets.update(super(cls, self).lc_secrets)  # type: ignore [arg-type]

        # Get latest values for kwargs if there is an attribute with same name
        lc_kwargs = {k: getattr(self, k, v) for k, v in self.lc_kwargs.items()}
        # Add additional attributes from lc_attributes
        lc_kwargs.update({k: getattr(self, k) for k in self.lc_attributes})

        return {
            "lc": 1,
            "type": "constructor",
            "id": [*self.lc_namespace, self.__class__.__name__],
            "kwargs": lc_kwargs if not secrets else replace_secrets(lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        return {
            "lc": 1,
            "type": "not_implemented",
            "id": [*self.lc_namespace, self.__class__.__name__],
        }


def replace_secrets(
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
