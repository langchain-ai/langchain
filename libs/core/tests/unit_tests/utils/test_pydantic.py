"""Test for some custom pydantic decorators."""

from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.pydantic import pre_init


def test_pre_init_decorator() -> None:
    class Foo(BaseModel):
        x: int = 5
        y: int

        @pre_init
        def validator(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            v["y"] = v["x"] + 1
            return v

    # Type ignore initialization b/c y is marked as required
    foo = Foo()  # type: ignore
    assert foo.y == 6
    foo = Foo(x=10)  # type: ignore
    assert foo.y == 11


def test_pre_init_decorator_with_more_defaults() -> None:
    class Foo(BaseModel):
        a: int = 1
        b: Optional[int] = None
        c: int = Field(default=2)
        d: int = Field(default_factory=lambda: 3)

        @pre_init
        def validator(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            assert v["a"] == 1
            assert v["b"] is None
            assert v["c"] == 2
            assert v["d"] == 3
            return v

    # Try to create an instance of Foo
    # nothing is required, but mypy can't track the default for `c`
    Foo()  # type: ignore


def test_with_aliases() -> None:
    class Foo(BaseModel):
        x: int = Field(default=1, alias="y")
        z: int

        class Config:
            allow_population_by_field_name = True

        @pre_init
        def validator(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            v["z"] = v["x"]
            return v

    # Based on defaults
    # z is required
    foo = Foo()  # type: ignore
    assert foo.x == 1
    assert foo.z == 1

    # Based on field name
    # z is required
    foo = Foo(x=2)  # type: ignore
    assert foo.x == 2
    assert foo.z == 2

    # Based on alias
    # z is required
    foo = Foo(y=2)  # type: ignore
    assert foo.x == 2
    assert foo.z == 2
