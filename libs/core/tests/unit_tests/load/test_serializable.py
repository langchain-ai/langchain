from typing import Dict

from pydantic import ConfigDict, Field

from langchain_core.load import Serializable, dumpd, load
from langchain_core.load.serializable import _is_field_useful


def test_simple_serialization() -> None:
    class Foo(Serializable):
        bar: int
        baz: str

    foo = Foo(bar=1, baz="hello")
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "lc": 1,
        "repr": "Foo(bar=1, baz='hello')",
        "type": "not_implemented",
    }


def test_simple_serialization_is_serializable() -> None:
    class Foo(Serializable):
        bar: int
        baz: str

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }


def test_simple_serialization_secret() -> None:
    """Test handling of secrets."""
    from pydantic import SecretStr

    from langchain_core.load import Serializable

    class Foo(Serializable):
        bar: int
        baz: str
        secret: SecretStr
        secret_2: str

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_secrets(self) -> Dict[str, str]:
            return {"secret": "MASKED_SECRET", "secret_2": "MASKED_SECRET_2"}

    foo = Foo(
        bar=1, baz="baz", secret=SecretStr("SUPER_SECRET"), secret_2="SUPER_SECRET"
    )
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {
            "bar": 1,
            "baz": "baz",
            "secret": {"id": ["MASKED_SECRET"], "lc": 1, "type": "secret"},
            "secret_2": {"id": ["MASKED_SECRET_2"], "lc": 1, "type": "secret"},
        },
        "lc": 1,
        "type": "constructor",
    }


def test__is_field_useful() -> None:
    class ArrayObj:
        def __bool__(self) -> bool:
            raise ValueError("Truthiness can't be determined")

        def __eq__(self, other: object) -> bool:
            return self  # type: ignore[return-value]

    class NonBoolObj:
        def __bool__(self) -> bool:
            raise ValueError("Truthiness can't be determined")

        def __eq__(self, other: object) -> bool:
            raise ValueError("Equality can't be determined")

    default_x = ArrayObj()
    default_y = NonBoolObj()

    class Foo(Serializable):
        x: ArrayObj = Field(default=default_x)
        y: NonBoolObj = Field(default=default_y)
        # Make sure works for fields without default.
        z: ArrayObj

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )

    foo = Foo(x=ArrayObj(), y=NonBoolObj(), z=ArrayObj())
    assert _is_field_useful(foo, "x", foo.x)
    assert _is_field_useful(foo, "y", foo.y)

    foo = Foo(x=default_x, y=default_y, z=ArrayObj())
    assert not _is_field_useful(foo, "x", foo.x)
    assert not _is_field_useful(foo, "y", foo.y)


class Foo(Serializable):
    bar: int
    baz: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_simple_deserialization() -> None:
    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    serialized_foo = dumpd(foo)
    assert serialized_foo == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }
    new_foo = load(serialized_foo, valid_namespaces=["tests"])
    assert new_foo == foo


class Foo2(Serializable):
    bar: int
    baz: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_simple_deserialization_with_additional_imports() -> None:
    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    serialized_foo = dumpd(foo)
    assert serialized_foo == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }
    new_foo = load(
        serialized_foo,
        valid_namespaces=["tests"],
        additional_import_mappings={
            ("tests", "unit_tests", "load", "test_serializable", "Foo"): (
                "tests",
                "unit_tests",
                "load",
                "test_serializable",
                "Foo2",
            )
        },
    )
    assert isinstance(new_foo, Foo2)
