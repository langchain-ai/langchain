"""Test for some custom pydantic decorators."""

import sys
import warnings
from typing import Annotated, Any, get_type_hints

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_core.utils.pydantic import (
    _create_subset_model_v2,
    create_model_v2,
    get_fields,
    is_basemodel_instance,
    is_basemodel_subclass,
    pre_init,
)
from tests.unit_tests.utils.pydantic_future_annotations import FutureModel


def test_pre_init_decorator() -> None:
    class Foo(BaseModel):
        x: int = 5
        y: int

        @pre_init
        def validator(cls, v: dict[str, Any]) -> dict[str, Any]:
            v["y"] = v["x"] + 1
            return v

    # Type ignore initialization b/c y is marked as required
    foo = Foo()  # type: ignore[call-arg]
    assert foo.y == 6
    foo = Foo(x=10)  # type: ignore[call-arg]
    assert foo.y == 11


def test_pre_init_decorator_with_more_defaults() -> None:
    class Foo(BaseModel):
        a: int = 1
        b: int | None = None
        c: int = Field(default=2)
        d: int = Field(default_factory=lambda: 3)

        @pre_init
        def validator(cls, v: dict[str, Any]) -> dict[str, Any]:
            assert v["a"] == 1
            assert v["b"] is None
            assert v["c"] == 2
            assert v["d"] == 3
            return v

    # Try to create an instance of Foo
    Foo()


def test_with_aliases() -> None:
    class Foo(BaseModel):
        x: int = Field(default=1, alias="y")
        z: int

        model_config = ConfigDict(
            populate_by_name=True,
        )

        @pre_init
        def validator(cls, v: dict[str, Any]) -> dict[str, Any]:
            v["z"] = v["x"]
            return v

    # Based on defaults
    # z is required
    foo = Foo()  # type: ignore[call-arg]
    assert foo.x == 1
    assert foo.z == 1

    # Based on field name
    # z is required
    foo = Foo(x=2)  # type: ignore[call-arg]
    assert foo.x == 2
    assert foo.z == 2

    # Based on alias
    # z is required
    foo = Foo(y=2)  # type: ignore[call-arg]
    assert foo.x == 2
    assert foo.z == 2


def test_is_basemodel_subclass() -> None:
    """Test pydantic."""
    assert is_basemodel_subclass(BaseModel)
    assert is_basemodel_subclass(BaseModelV1)


def test_is_basemodel_instance() -> None:
    """Test pydantic."""

    class Foo(BaseModel):
        x: int

    assert is_basemodel_instance(Foo(x=5))

    class Bar(BaseModelV1):
        x: int

    assert is_basemodel_instance(Bar(x=5))


def test_with_field_metadata() -> None:
    """Test pydantic with field metadata."""

    class Foo(BaseModel):
        x: list[int] = Field(
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


def test_create_subset_model_v2_resolves_string_annotations() -> None:
    """Test that lazy string annotations are resolved on the subset model."""
    subset_model = _create_subset_model_v2("Sub", FutureModel, ["metadata"])
    assert subset_model.__annotations__ == {"metadata": dict[str, Any] | None}


def test_create_subset_model_v2_preserves_annotated_extras() -> None:
    """Test that `Annotated` metadata survives string annotation resolution."""
    subset_model = _create_subset_model_v2("Sub", FutureModel, ["tagged"])
    assert subset_model.__annotations__ == {"tagged": Annotated[dict, "extra"] | None}


class LocalRegistry:
    """Module-level shadow without `Inner` for the fallback test below."""


def test_create_subset_model_v2_unresolvable_annotations_fall_back() -> None:
    """Test the fallback for annotations `get_type_hints` cannot resolve.

    Models defined inside function bodies can hold forward references to other
    function-local names. Pydantic resolves those from the enclosing frame,
    but `typing.get_type_hints` only sees the module namespace and raises,
    for example `NameError` for an unknown name or `AttributeError` for an
    attribute of a shadowed name. Subset model creation must not raise for
    such models; the unresolvable annotation is copied as the raw string
    instead.
    """

    class LocalModel(BaseModel):
        x: "LocalDep | None" = None

    class LocalDep(BaseModel):
        y: int = 0

    LocalModel.model_rebuild()

    with pytest.raises(NameError):
        get_type_hints(LocalModel)

    subset_model = _create_subset_model_v2("Sub", LocalModel, ["x"])
    assert subset_model.__annotations__ == {"x": "LocalDep | None"}

    class LocalRegistry:
        class Inner(BaseModel):
            y: int = 0

    class ShadowModel(BaseModel):
        x: "LocalRegistry.Inner | None" = None

    # `get_type_hints` resolves `LocalRegistry` to the module-level class of
    # the same name above, which has no `Inner`.
    with pytest.raises(AttributeError):
        get_type_hints(ShadowModel)

    subset_model = _create_subset_model_v2("Sub", ShadowModel, ["x"])
    assert subset_model.__annotations__ == {"x": "LocalRegistry.Inner | None"}


def test_fields_pydantic_v2_proper() -> None:
    class Foo(BaseModel):
        x: int

    fields = get_fields(Foo)
    assert fields == {"x": Foo.model_fields["x"]}


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="pydantic.v1 namespace not supported with Python 3.14+",
)
def test_fields_pydantic_v1_from_2() -> None:
    class Foo(BaseModelV1):
        x: int

    fields = get_fields(Foo)
    assert fields == {"x": Foo.__fields__["x"]}


def test_create_model_v2() -> None:
    """Test that create model v2 works as expected."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        foo = create_model_v2("Foo", field_definitions={"a": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    # schema is used by pydantic, but OK to re-use
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        foo = create_model_v2("Foo", field_definitions={"schema": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    # From protected namespaces, but definitely OK to use.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        foo = create_model_v2("Foo", field_definitions={"model_id": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        # Verify that we can use non-English characters
        field_name = "もしもし"
        foo = create_model_v2("Foo", field_definitions={field_name: (int, None)})
        foo.model_json_schema()

    assert list(record) == []


def test_create_subset_model_v2_preserves_default_factory() -> None:
    """Fields with default_factory should not be marked as required."""

    class Original(BaseModel):
        required_field: str
        names: list[str] = Field(default_factory=list, description="Some names")
        mapping: dict[str, int] = Field(default_factory=dict, description="A mapping")

    subset = _create_subset_model_v2(
        "Subset",
        Original,
        ["required_field", "names", "mapping"],
    )
    schema = subset.model_json_schema()
    assert schema.get("required") == ["required_field"]
    assert "names" not in schema.get("required", [])
    assert "mapping" not in schema.get("required", [])
