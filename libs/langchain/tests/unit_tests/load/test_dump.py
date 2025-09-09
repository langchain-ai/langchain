"""Test for Serializable base class."""

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.load.dump import dumps
from langchain_core.load.serializable import Serializable
from pydantic import ConfigDict, Field, model_validator


class Person(Serializable):
    secret: str

    you_can_see_me: str = "hello"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"secret": "SECRET"}

    @property
    def lc_attributes(self) -> dict[str, str]:
        return {"you_can_see_me": self.you_can_see_me}


class SpecialPerson(Person):
    another_secret: str

    another_visible: str = "bye"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["my", "special", "namespace"]

    # Gets merged with parent class's secrets
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"another_secret": "ANOTHER_SECRET"}

    # Gets merged with parent class's attributes
    @property
    def lc_attributes(self) -> dict[str, str]:
        return {"another_visible": self.another_visible}


class NotSerializable:
    pass


def test_person(snapshot: Any) -> None:
    p = Person(secret="parrot party")  # noqa: S106
    assert dumps(p, pretty=True) == snapshot
    sp = SpecialPerson(another_secret="Wooo", secret="Hmm")  # noqa: S106
    assert dumps(sp, pretty=True) == snapshot
    assert Person.lc_id() == ["tests", "unit_tests", "load", "test_dump", "Person"]
    assert SpecialPerson.lc_id() == ["my", "special", "namespace", "SpecialPerson"]


def test_typeerror() -> None:
    assert (
        dumps({(1, 2): 3})
        == """{"lc": 1, "type": "not_implemented", "id": ["builtins", "dict"], "repr": "{(1, 2): 3}"}"""  # noqa: E501
    )


def test_person_with_kwargs(snapshot: Any) -> None:
    person = Person(secret="parrot party")  # noqa: S106
    assert dumps(person, separators=(",", ":")) == snapshot


def test_person_with_invalid_kwargs() -> None:
    person = Person(secret="parrot party")  # noqa: S106
    with pytest.raises(TypeError):
        dumps(person, invalid_kwarg="hello")


class TestClass(Serializable):
    my_favorite_secret: str = Field(alias="my_favorite_secret_alias")
    my_other_secret: str = Field()

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def get_from_env(cls, values: dict) -> Any:
        """Get the values from the environment."""
        if "my_favorite_secret" not in values:
            values["my_favorite_secret"] = os.getenv("MY_FAVORITE_SECRET")
        if "my_other_secret" not in values:
            values["my_other_secret"] = os.getenv("MY_OTHER_SECRET")
        return values

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["my", "special", "namespace"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {
            "my_favorite_secret": "MY_FAVORITE_SECRET",
            "my_other_secret": "MY_OTHER_SECRET",
        }


def test_aliases_hidden() -> None:
    test_class = TestClass(
        my_favorite_secret="hello",  # noqa: S106
        my_other_secret="world",  # noqa: S106
    )
    dumped = json.loads(dumps(test_class, pretty=True))
    expected_dump = {
        "lc": 1,
        "type": "constructor",
        "id": ["my", "special", "namespace", "TestClass"],
        "kwargs": {
            "my_favorite_secret": {
                "lc": 1,
                "type": "secret",
                "id": ["MY_FAVORITE_SECRET"],
            },
            "my_other_secret": {"lc": 1, "type": "secret", "id": ["MY_OTHER_SECRET"]},
        },
    }
    assert dumped == expected_dump
    # Check while patching the os environment
    with patch.dict(
        os.environ,
        {"MY_FAVORITE_SECRET": "hello", "MY_OTHER_SECRET": "world"},
    ):
        test_class = TestClass()  # type: ignore[call-arg]
        dumped = json.loads(dumps(test_class, pretty=True))

    # Check by alias
    test_class = TestClass(  # type: ignore[call-arg]
        my_favorite_secret_alias="hello",  # noqa: S106
        my_other_secret="parrot party",  # noqa: S106
    )
    dumped = json.loads(dumps(test_class, pretty=True))
    expected_dump = {
        "lc": 1,
        "type": "constructor",
        "id": ["my", "special", "namespace", "TestClass"],
        "kwargs": {
            "my_favorite_secret": {
                "lc": 1,
                "type": "secret",
                "id": ["MY_FAVORITE_SECRET"],
            },
            "my_other_secret": {"lc": 1, "type": "secret", "id": ["MY_OTHER_SECRET"]},
        },
    }
    assert dumped == expected_dump
