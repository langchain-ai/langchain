from langchain_core.pydantic_v1 import BaseModel, v1
from tests.unit_tests.pydantic_utils import _schema


def test_schemas() -> None:
    """Test schema remapping for the two pydantic versions."""

    class Bar(BaseModel):
        baz: int

    class Foo(BaseModel):
        bar: Bar

    schema_2 = _schema(Foo)

    class Bar(v1.BaseModel):
        baz: int

    class Foo(v1.BaseModel):
        bar: Bar

    schema_1 = _schema(Foo)

    assert schema_1 == schema_2
