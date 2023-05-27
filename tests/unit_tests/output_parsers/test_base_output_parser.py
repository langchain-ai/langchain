"""Test the BaseOutputParser class and its sub-classes."""
from abc import ABC
from typing import List, Optional, Set, Type

import pytest

from langchain.schema import BaseOutputParser


def non_abstract_subclasses(
    cls: Type[ABC], to_skip: Optional[Set] = None
) -> List[Type]:
    """Recursively find all non-abstract subclasses of a class."""
    _to_skip = to_skip or set()
    subclasses = []
    for subclass in cls.__subclasses__():
        if not getattr(subclass, "__abstractmethods__", None):
            if subclass.__name__ not in _to_skip:
                subclasses.append(subclass)
        subclasses.extend(non_abstract_subclasses(subclass, to_skip=_to_skip))
    return subclasses


_PARSERS_TO_SKIP = {"FakeOutputParser", "BaseOutputParser"}
_NON_ABSTRACT_PARSERS = non_abstract_subclasses(
    BaseOutputParser, to_skip=_PARSERS_TO_SKIP
)


@pytest.mark.parametrize("cls", _NON_ABSTRACT_PARSERS)
def test_subclass_implements_type(cls: Type[BaseOutputParser]) -> None:
    try:
        cls._type
    except NotImplementedError:
        pytest.fail(f"_type property is not implemented in class {cls.__name__}")


def test_all_subclasses_implement_unique_type() -> None:
    types = []
    for cls in _NON_ABSTRACT_PARSERS:
        try:
            types.append(cls._type)
        except NotImplementedError:
            # This is handled in the previous test
            pass
    dups = set([t for t in types if types.count(t) > 1])
    assert not dups, f"Duplicate types: {dups}"
