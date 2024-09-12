"""Test for some custom pydantic decorators."""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import ConfigDict

from langchain_core.utils.pydantic import (
    PYDANTIC_MAJOR_VERSION,
    _create_subset_model_v2,
    create_model_v2,
    get_fields,
    is_basemodel_instance,
    is_basemodel_subclass,
    pre_init,
)


def test_pre_init_decorator() -> None:
    from pydantic import BaseModel

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
    from pydantic import BaseModel, Field

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
    from pydantic import BaseModel, Field

    class Foo(BaseModel):
        x: int = Field(default=1, alias="y")
        z: int

        model_config = ConfigDict(
            populate_by_name=True,
        )

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


def test_is_basemodel_subclass() -> None:
    """Test pydantic."""
    if PYDANTIC_MAJOR_VERSION == 1:
        from pydantic import BaseModel as BaseModelV1Proper

        assert is_basemodel_subclass(BaseModelV1Proper)
    elif PYDANTIC_MAJOR_VERSION == 2:
        from pydantic import BaseModel as BaseModelV2
        from pydantic.v1 import BaseModel as BaseModelV1

        assert is_basemodel_subclass(BaseModelV2)

        assert is_basemodel_subclass(BaseModelV1)
    else:
        raise ValueError(f"Unsupported Pydantic version: {PYDANTIC_MAJOR_VERSION}")


def test_is_basemodel_instance() -> None:
    """Test pydantic."""
    if PYDANTIC_MAJOR_VERSION == 1:
        from pydantic import BaseModel as BaseModelV1Proper

        class FooV1(BaseModelV1Proper):
            x: int

        assert is_basemodel_instance(FooV1(x=5))
    elif PYDANTIC_MAJOR_VERSION == 2:
        from pydantic import BaseModel as BaseModelV2
        from pydantic.v1 import BaseModel as BaseModelV1

        class Foo(BaseModelV2):
            x: int

        assert is_basemodel_instance(Foo(x=5))

        class Bar(BaseModelV1):
            x: int

        assert is_basemodel_instance(Bar(x=5))
    else:
        raise ValueError(f"Unsupported Pydantic version: {PYDANTIC_MAJOR_VERSION}")


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Only tests Pydantic v2")
def test_with_field_metadata() -> None:
    """Test pydantic with field metadata"""
    from pydantic import BaseModel as BaseModelV2
    from pydantic import Field as FieldV2

    class Foo(BaseModelV2):
        x: List[int] = FieldV2(
            description="List of integers", min_length=10, max_length=15
        )

    subset_model = _create_subset_model_v2("Foo", Foo, ["x"])
    assert subset_model.model_json_schema() == {
        "properties": {
            "x": {
                "description": "List of integers",
                "items": {"type": "integer"},
                "maxItems": 15,
                "minItems": 10,
                "title": "X",
                "type": "array",
            }
        },
        "required": ["x"],
        "title": "Foo",
        "type": "object",
    }


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 1, reason="Only tests Pydantic v1")
def test_fields_pydantic_v1() -> None:
    from pydantic import BaseModel

    class Foo(BaseModel):
        x: int

    fields = get_fields(Foo)
    assert fields == {"x": Foo.model_fields["x"]}  # type: ignore[index]


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Only tests Pydantic v2")
def test_fields_pydantic_v2_proper() -> None:
    from pydantic import BaseModel

    class Foo(BaseModel):
        x: int

    fields = get_fields(Foo)
    assert fields == {"x": Foo.model_fields["x"]}


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Only tests Pydantic v2")
def test_fields_pydantic_v1_from_2() -> None:
    from pydantic.v1 import BaseModel

    class Foo(BaseModel):
        x: int

    fields = get_fields(Foo)
    assert fields == {"x": Foo.__fields__["x"]}


def test_create_model_v2() -> None:
    """Test that create model v2 works as expected."""

    with pytest.warns(None) as record:  # type: ignore
        foo = create_model_v2("Foo", field_definitions={"a": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    # schema is used by pydantic, but OK to re-use
    with pytest.warns(None) as record:  # type: ignore
        foo = create_model_v2("Foo", field_definitions={"schema": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    # From protected namespaces, but definitely OK to use.
    with pytest.warns(None) as record:  # type: ignore
        foo = create_model_v2("Foo", field_definitions={"model_id": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    with pytest.warns(None) as record:  # type: ignore
        # Verify that we can use non-English characters
        field_name = "もしもし"
        foo = create_model_v2("Foo", field_definitions={field_name: (int, None)})
        foo.model_json_schema()

    assert list(record) == []
