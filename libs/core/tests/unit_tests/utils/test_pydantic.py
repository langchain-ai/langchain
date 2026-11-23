"""Test for some custom pydantic decorators."""

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    PydanticDeprecationWarning,
    RootModel,
)
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic_core import core_schema

from langchain_core.utils.pydantic import (
    _create_subset_model_v2,
    create_model_v2,
    get_fields,
    is_basemodel_instance,
    is_basemodel_subclass,
    model_json_schema,
    model_validate,
    pre_init,
)


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


def test_model_json_schema_v2() -> None:
    class Foo(BaseModel):
        x: int

    assert model_json_schema(Foo) == Foo.model_json_schema()


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="pydantic.v1 namespace not supported with Python 3.14+",
)
def test_model_json_schema_v1() -> None:
    class Foo(BaseModelV1):
        x: int

    assert model_json_schema(Foo) == Foo.schema()


def test_model_json_schema_non_model() -> None:
    with pytest.raises(TypeError, match="Expected a Pydantic model"):
        model_json_schema(dict)  # type: ignore[arg-type]


def test_model_validate_v2() -> None:
    class Foo(BaseModel):
        x: int

    result = model_validate(Foo, {"x": 1})
    assert isinstance(result, Foo)
    assert result.x == 1


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="pydantic.v1 namespace not supported with Python 3.14+",
)
def test_model_validate_v1() -> None:
    class Foo(BaseModelV1):
        x: int

    result = model_validate(Foo, {"x": 1})
    assert isinstance(result, Foo)
    assert result.x == 1


def test_model_validate_non_model() -> None:
    with pytest.raises(TypeError, match="Expected a Pydantic model"):
        model_validate(dict, {"x": 1})  # type: ignore[arg-type]


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


@pytest.mark.parametrize(
    "schema_kind", ["fields", "unhashable_fields", "root", "unhashable_root"]
)
def test_create_model_v2_preserves_concurrent_warning_filters(schema_kind: str) -> None:
    entered = Event()
    release = Event()

    class PausingType:
        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source: Any, _handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            entered.set()
            assert release.wait(timeout=10)
            return core_schema.list_schema(core_schema.int_schema())

    default: Any = [] if schema_kind.startswith("unhashable") else ...
    kwargs = (
        {"root": (PausingType, default)}
        if schema_kind.endswith("root")
        else {"field_definitions": {"model_name": (PausingType, default)}}
    )
    with warnings.catch_warnings(), ThreadPoolExecutor(max_workers=1) as executor:
        warnings.simplefilter("error", DeprecationWarning)
        future = executor.submit(create_model_v2, "ConcurrentSchema", **kwargs)
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(DeprecationWarning, match="unrelated deprecation"):
                warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)
            warnings.simplefilter("error", UserWarning)
        finally:
            release.set()
        model = future.result(timeout=10)
        with pytest.raises(UserWarning, match="concurrent filter update"):
            warnings.warn("concurrent filter update", UserWarning, stacklevel=1)
    assert model.__name__ == "ConcurrentSchema"
    assert model.model_json_schema()["title"] == "ConcurrentSchema"


@pytest.mark.parametrize("default", [..., []])
def test_create_model_v2_preserves_field_names_and_aliases(default: Any) -> None:
    names = ("model_name", "model_dump", "schema", "_private", "もしもし")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = create_model_v2(
            "FieldNames", field_definitions=dict.fromkeys(names, (list[int], default))
        )
        values = {name: [1] for name in names}
        assert model.model_validate(values).model_dump(by_alias=True) == values
        assert set(model.model_json_schema()["properties"]) == set(values)


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="pydantic.v1 namespace not supported with Python 3.14+",
)
def test_create_model_v2_preserves_v1_root_warnings() -> None:
    class LegacyModel(BaseModelV1):
        value: int

    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        type(
            "ExpectedLegacyRoot",
            (RootModel,),
            {"__annotations__": {"root": LegacyModel}},
        )
    with warnings.catch_warnings(record=True) as actual:
        warnings.simplefilter("always")
        model = create_model_v2("LegacyRoot", root=LegacyModel)
    assert [(warning.category, str(warning.message)) for warning in actual] == [
        (warning.category, str(warning.message))
        for warning in expected
        if not issubclass(warning.category, PydanticDeprecationWarning)
    ]
    assert model.model_fields["root"].annotation is LegacyModel


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
