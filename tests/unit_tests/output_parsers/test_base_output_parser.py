"""Test the BaseOutputParser class and its sub-classes."""
from abc import ABC
from typing import List, Type
from unittest.mock import MagicMock

import pytest

from langchain.schema import BaseOutputParser

_FAKE_PARSERS = {"FakeOutputParser"}


def non_abstract_subclasses(cls: Type[ABC]) -> List[Type]:
    """Recursively find all non-abstract subclasses of a class."""
    subclasses = []
    for subclass in cls.__subclasses__():
        if not getattr(subclass, "__abstractmethods__", None):
            if subclass.__name__ not in _FAKE_PARSERS:
                subclasses.append(subclass)
        subclasses.extend(non_abstract_subclasses(subclass))
    return subclasses


@pytest.mark.parametrize("cls", non_abstract_subclasses(BaseOutputParser))
def test_all_subclasses_implement_type(cls: Type[BaseOutputParser]) -> None:
    try:
        # Most parsers just return a string. MagicMock lets
        # the parsers that wrap another parsers slide by
        cls._type.fget(MagicMock())  # type: ignore
    except NotImplementedError:
        pytest.fail(f"_type property is not implemented in class {cls.__name__}")


def test_all_subclasses_implement_unique_type() -> None:
    types = [cls._type for cls in non_abstract_subclasses(BaseOutputParser)]
    assert len(types) == len(set(types))
