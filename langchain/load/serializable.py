from abc import ABC, abstractmethod
from typing import TypedDict, Literal, Any

from pydantic import BaseModel


class BaseSerialized(TypedDict):
    lc: int
    type: str
    id: list[str]


class SerializedConstructor(BaseSerialized):
    type: Literal["constructor"]
    kwargs: dict[str, Any]


class SerializedSecret(BaseSerialized):
    type: Literal["secret"]


class SerializedNotImplemented(BaseSerialized):
    type: Literal["not_implemented"]


class Serializable(BaseModel, ABC):
    @property
    def lc_namespace(self) -> list[str]:
        """
        Return the namespace of the langchain object.
        eg. ["langchain", "llms", "openai"]
        """
        return self.__class__.__module__.split(".")

    @property
    def lc_secrets(self) -> dict[str, str]:
        """
        Return a map of constructor argument names to secret ids.
        eg. {"openai_api_key": "OPENAI_API_KEY"}
        """
        return dict()

    lc_kwargs: dict[str, Any] = dict()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lc_kwargs = kwargs

    def to_json(self) -> SerializedConstructor:
        secrets = dict()

        # Merge the lc_secrets from every class in the MRO
        for cls in self.__class__.mro():
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            secrets.update(super(cls, self).lc_secrets)

        return {
            "lc": 1,
            "type": "constructor",
            "id": [*self.lc_namespace, self.__class__.__name__],
            "kwargs": self.lc_kwargs
            if not secrets
            else replace_secrets(self.lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        return {
            "lc": 1,
            "type": "not_implemented",
            "id": [*self.lc_namespace, self.__class__.__name__],
        }


def replace_secrets(root: dict[Any, Any], secrets_map: dict[str, str]):
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
