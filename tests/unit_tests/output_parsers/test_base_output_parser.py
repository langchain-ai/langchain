"""Test the BaseOutputParser class and its sub-classes."""
from abc import ABC
from typing import List, Set, Type, Optional
from unittest.mock import MagicMock

import pytest

from langchain.schema import BaseOutputParser

_PARSERS_TO_SKIP = {"FakeOutputParser", "BaseOutputParser"}


def non_abstract_subclasses(cls: Type[ABC], to_skip: Optional[Set]=None) -> List[Type]:
    """Recursively find all non-abstract subclasses of a class."""
    subclasses = []
    for subclass in cls.__subclasses__():
        if not getattr(subclass, "__abstractmethods__", None):
            if subclass.__name__ not in to_skip:
                subclasses.append(subclass)
        subclasses.extend(non_abstract_subclasses(subclass, to_skip))
    return subclasses


@pytest.mark.parametrize("cls", non_abstract_subclasses(BaseOutputParser, _PARSERS_TO_SKIP))
def test_subclass_implements_type(cls: Type[BaseOutputParser]) -> None:
    try:
        cls._type
    except NotImplementedError:
        pytest.fail(f"_type property is not implemented in class {cls.__name__}")


def test_all_subclasses_implement_unique_type() -> None:
    types = []
    for cls in non_abstract_subclasses(BaseOutputParser, _PARSERS_TO_SKIP):
        try:
            types.append(cls._type)
        except NotImplementedError:
            # This is handled in the previous test
            pass
    dups = set([t for t in types if types.count(t) > 1])
    assert not dups, f"Duplicate types: {dups}"
