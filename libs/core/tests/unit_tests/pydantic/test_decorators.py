"""Test for some custom pydantic decorators."""
from langchain_core.pydantic_v1 import BaseModel

from langchain_core.utils.pydantic import pre_init


class Foo(BaseModel):
    x: int = 5
    y: int

    @pre_init
    def validator(cls, v):
        v["y"] = v["x"] + 1
        return v


def test_pre_init_decorator() -> None:
    foo = Foo()
    assert foo.y == 6
    foo = Foo(x=10)
    assert foo.y == 11
