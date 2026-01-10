import inspect
import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from langchain_core._api.deprecation import (
    deprecated,
    rename_parameter,
    warn_deprecated,
)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        (
            {
                "since": "1.0.0",
                "name": "OldClass",
                "alternative": "NewClass",
                "pending": True,
                "obj_type": "class",
            },
            "The class `OldClass` will be deprecated in a future version. Use NewClass "
            "instead.",
        ),
        (
            {
                "since": "2.0.0",
                "message": "This is a custom message",
                "name": "FunctionA",
                "alternative": "",
                "pending": True,
                "obj_type": "",
                "addendum": "",
                "removal": "",
            },
            "This is a custom message",
        ),
        (
            {
                "since": "1.5.0",
                "message": "",
                "name": "SomeFunction",
                "alternative": "",
                "pending": False,
                "obj_type": "",
                "addendum": "Please migrate your code.",
                "removal": "2.5.0",
            },
            "`SomeFunction` was deprecated in LangChain 1.5.0 and will be "
            "removed in 2.5.0 Please migrate your code.",
        ),
    ],
)
def test_warn_deprecated(kwargs: dict[str, Any], expected_message: str) -> None:
    """Test warn deprecated."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        warn_deprecated(**kwargs)

        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == expected_message


def test_undefined_deprecation_schedule() -> None:
    """This test is expected to fail until we defined a deprecation schedule."""
    with pytest.raises(NotImplementedError):
        warn_deprecated("1.0.0", pending=False)


@deprecated(since="2.0.0", removal="3.0.0", pending=False)
def deprecated_function() -> str:
    """Original doc."""
    return "This is a deprecated function."


@deprecated(since="2.0.0", removal="3.0.0", pending=False)
async def deprecated_async_function() -> str:
    """Original doc."""
    return "This is a deprecated async function."


class ClassWithDeprecatedMethods:
    def __init__(self) -> None:
        """Original doc."""

    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_method(self) -> str:
        """Original doc."""
        return "This is a deprecated method."

    @deprecated(since="2.0.0", removal="3.0.0")
    async def deprecated_async_method(self) -> str:
        """Original doc."""
        return "This is a deprecated async method."

    @classmethod
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_classmethod(cls) -> str:
        """Original doc."""
        return "This is a deprecated classmethod."

    @staticmethod
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_staticmethod() -> str:
        """Original doc."""
        return "This is a deprecated staticmethod."

    @property
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_property(self) -> str:
        """Original doc."""
        return "This is a deprecated property."


def test_deprecated_function() -> None:
    """Test deprecated function."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert deprecated_function() == "This is a deprecated function."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The function `deprecated_function` was deprecated in LangChain 2.0.0 "
            "and will be removed in 3.0.0"
        )

        doc = deprecated_function.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")

    assert not inspect.iscoroutinefunction(deprecated_function)


async def test_deprecated_async_function() -> None:
    """Test deprecated async function."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert (
            await deprecated_async_function() == "This is a deprecated async function."
        )
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The function `deprecated_async_function` was deprecated "
            "in LangChain 2.0.0 and will be removed in 3.0.0"
        )

        doc = deprecated_function.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")

    assert inspect.iscoroutinefunction(deprecated_async_function)


def test_deprecated_method() -> None:
    """Test deprecated method."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        obj = ClassWithDeprecatedMethods()
        assert obj.deprecated_method() == "This is a deprecated method."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `ClassWithDeprecatedMethods.deprecated_method` was deprecated"
            " in tests 2.0.0 and will be removed in 3.0.0"
        )

        doc = obj.deprecated_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")

    assert not inspect.iscoroutinefunction(obj.deprecated_method)


async def test_deprecated_async_method() -> None:
    """Test deprecated async method."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        obj = ClassWithDeprecatedMethods()
        assert (
            await obj.deprecated_async_method() == "This is a deprecated async method."
        )
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `ClassWithDeprecatedMethods.deprecated_async_method` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )

        doc = obj.deprecated_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")

    assert inspect.iscoroutinefunction(obj.deprecated_async_method)


def test_deprecated_classmethod() -> None:
    """Test deprecated classmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        ClassWithDeprecatedMethods.deprecated_classmethod()
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `ClassWithDeprecatedMethods.deprecated_classmethod` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )

        doc = ClassWithDeprecatedMethods.deprecated_classmethod.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")


def test_deprecated_staticmethod() -> None:
    """Test deprecated staticmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert (
            ClassWithDeprecatedMethods.deprecated_staticmethod()
            == "This is a deprecated staticmethod."
        )
        assert len(warning_list) == 1
        warning = warning_list[0].message

        assert str(warning) == (
            "The method `ClassWithDeprecatedMethods.deprecated_staticmethod` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )
        doc = ClassWithDeprecatedMethods.deprecated_staticmethod.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")


def test_deprecated_property() -> None:
    """Test deprecated staticmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = ClassWithDeprecatedMethods()
        assert obj.deprecated_property == "This is a deprecated property."

        assert len(warning_list) == 1
        warning = warning_list[0].message

        assert str(warning) == (
            "The method `ClassWithDeprecatedMethods.deprecated_property` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )
        doc = ClassWithDeprecatedMethods.deprecated_property.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")


def test_whole_class_deprecation() -> None:
    """Test whole class deprecation."""

    # Test whole class deprecation
    @deprecated(since="2.0.0", removal="3.0.0")
    class DeprecatedClass:
        def __init__(self) -> None:
            """Original doc."""

        @deprecated(since="2.0.0", removal="3.0.0")
        def deprecated_method(self) -> str:
            """Original doc."""
            return "This is a deprecated method."

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = DeprecatedClass()
        assert obj.deprecated_method() == "This is a deprecated method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class `test_whole_class_deprecation.<locals>.DeprecatedClass` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The method `test_whole_class_deprecation.<locals>.DeprecatedClass."
            "deprecated_method` was deprecated in "
            "tests 2.0.0 and will be removed in 3.0.0"
        )
        # [*Deprecated*] should be inserted only once:
        if obj.__doc__ is not None:
            assert obj.__doc__.count("!!! deprecated") == 1


def test_whole_class_inherited_deprecation() -> None:
    """Test whole class deprecation for inherited class.

    The original version of deprecation decorator created duplicates with
    '[*Deprecated*]'.
    """

    # Test whole class deprecation
    @deprecated(since="2.0.0", removal="3.0.0")
    class DeprecatedClass:
        def __init__(self) -> None:
            """Original doc."""

        @deprecated(since="2.0.0", removal="3.0.0")
        def deprecated_method(self) -> str:
            """Original doc."""
            return "This is a deprecated method."

    @deprecated(since="2.2.0", removal="3.2.0")
    class InheritedDeprecatedClass(DeprecatedClass):
        """Inherited deprecated class."""

        def __init__(self) -> None:
            """Original doc."""

        @deprecated(since="2.2.0", removal="3.2.0")
        def deprecated_method(self) -> str:
            """Original doc."""
            return "This is a deprecated method."

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = DeprecatedClass()
        assert obj.deprecated_method() == "This is a deprecated method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class `test_whole_class_inherited_deprecation.<locals>."
            "DeprecatedClass` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The method `test_whole_class_inherited_deprecation.<locals>."
            "DeprecatedClass.deprecated_method` was deprecated in "
            "tests 2.0.0 and will be removed in 3.0.0"
        )
        # if [*Deprecated*] was inserted only once:
        if obj.__doc__ is not None:
            assert obj.__doc__.count("!!! deprecated") == 1

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = InheritedDeprecatedClass()
        assert obj.deprecated_method() == "This is a deprecated method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class "
            "`test_whole_class_inherited_deprecation.<locals>.InheritedDeprecatedClass`"
            " was deprecated in tests 2.2.0 and will be removed in 3.2.0"
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The method `test_whole_class_inherited_deprecation.<locals>."
            "InheritedDeprecatedClass.deprecated_method` was deprecated in "
            "tests 2.2.0 and will be removed in 3.2.0"
        )
        # if [*Deprecated*] was inserted only once:
        if obj.__doc__ is not None:
            assert obj.__doc__.count("!!! deprecated") == 1
            assert "!!! deprecated" in obj.__doc__


# Tests with pydantic models
class MyModel(BaseModel):
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_method(self) -> str:
        """Original doc."""
        return "This is a deprecated method."


def test_deprecated_method_pydantic() -> None:
    """Test deprecated method."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        obj = MyModel()
        assert obj.deprecated_method() == "This is a deprecated method."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `MyModel.deprecated_method` was deprecated in "
            "tests 2.0.0 and will be removed in 3.0.0"
        )

        doc = obj.deprecated_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("!!! deprecated")


def test_raise_error_for_bad_decorator() -> None:
    """Verify that errors raised on init rather than on use."""
    # Should not specify both `alternative` and `alternative_import`
    with pytest.raises(
        ValueError, match="Cannot specify both alternative and alternative_import"
    ):

        @deprecated(since="2.0.0", alternative="NewClass", alternative_import="hello")
        def deprecated_function() -> str:
            """Original doc."""
            return "This is a deprecated function."


def test_rename_parameter() -> None:
    """Test rename parameter."""

    @rename_parameter(since="2.0.0", removal="3.0.0", old="old_name", new="new_name")
    def foo(new_name: str) -> str:
        """Original doc."""
        return new_name

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert foo(old_name="hello") == "hello"  # type: ignore[call-arg]
        assert len(warning_list) == 1

        assert foo(new_name="hello") == "hello"
        assert foo("hello") == "hello"
        assert foo.__doc__ == "Original doc."
        with pytest.raises(TypeError):
            foo(meow="hello")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            assert foo("hello", old_name="hello")  # type: ignore[call-arg]

        with pytest.raises(TypeError):
            assert foo(old_name="goodbye", new_name="hello")  # type: ignore[call-arg]


async def test_rename_parameter_for_async_func() -> None:
    """Test rename parameter."""

    @rename_parameter(since="2.0.0", removal="3.0.0", old="old_name", new="new_name")
    async def foo(new_name: str) -> str:
        """Original doc."""
        return new_name

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert await foo(old_name="hello") == "hello"  # type: ignore[call-arg]
        assert len(warning_list) == 1
        assert await foo(new_name="hello") == "hello"
        assert await foo("hello") == "hello"
        assert foo.__doc__ == "Original doc."
        with pytest.raises(TypeError):
            await foo(meow="hello")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            assert await foo("hello", old_name="hello")  # type: ignore[call-arg]

        with pytest.raises(TypeError):
            assert await foo(old_name="a", new_name="hello")  # type: ignore[call-arg]


def test_rename_parameter_method() -> None:
    """Test that it works for a method."""

    class Foo:
        @rename_parameter(
            since="2.0.0", removal="3.0.0", old="old_name", new="new_name"
        )
        def a(self, new_name: str) -> str:
            return new_name

    foo = Foo()

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert foo.a(old_name="hello") == "hello"  # type: ignore[call-arg]
        assert len(warning_list) == 1
        assert str(warning_list[0].message) == (
            "The parameter `old_name` of `a` was deprecated in 2.0.0 and will be "
            "removed "
            "in 3.0.0 Use `new_name` instead."
        )

        assert foo.a(new_name="hello") == "hello"
        assert foo.a("hello") == "hello"

        with pytest.raises(TypeError):
            foo.a(meow="hello")  # type: ignore[call-arg]

        with pytest.raises(TypeError):
            assert foo.a("hello", old_name="hello")  # type: ignore[call-arg]


# Tests for PEP 702 __deprecated__ attribute


def test_deprecated_function_has_pep702_attribute() -> None:
    """Test that deprecated functions have `__deprecated__` attribute."""

    @deprecated(since="2.0.0", removal="3.0.0", alternative="new_function")
    def old_function() -> str:
        """Original doc."""
        return "old"

    assert hasattr(old_function, "__deprecated__")
    assert old_function.__deprecated__ == "Use new_function instead."


def test_deprecated_function_with_alternative_import_has_pep702_attribute() -> None:
    """Test `__deprecated__` with `alternative_import`."""

    @deprecated(
        since="2.0.0", removal="3.0.0", alternative_import="new_module.new_function"
    )
    def old_function() -> str:
        """Original doc."""
        return "old"

    assert hasattr(old_function, "__deprecated__")
    assert old_function.__deprecated__ == "Use new_module.new_function instead."


def test_deprecated_function_without_alternative_has_pep702_attribute() -> None:
    """Test `__deprecated__` without alternative shows `'Deprecated.'`."""

    @deprecated(since="2.0.0", removal="3.0.0")
    def old_function() -> str:
        """Original doc."""
        return "old"

    assert hasattr(old_function, "__deprecated__")
    assert old_function.__deprecated__ == "Deprecated."


def test_deprecated_class_has_pep702_attribute() -> None:
    """Test that deprecated classes have `__deprecated__` attribute (PEP 702)."""

    @deprecated(since="2.0.0", removal="3.0.0", alternative="NewClass")
    class OldClass:
        def __init__(self) -> None:
            """Original doc."""

    assert hasattr(OldClass, "__deprecated__")
    assert OldClass.__deprecated__ == "Use NewClass instead."


def test_deprecated_class_without_alternative_has_pep702_attribute() -> None:
    """Test `__deprecated__` on class without alternative."""

    @deprecated(since="2.0.0", removal="3.0.0")
    class OldClass:
        def __init__(self) -> None:
            """Original doc."""

    assert hasattr(OldClass, "__deprecated__")
    assert OldClass.__deprecated__ == "Deprecated."


def test_deprecated_property_has_pep702_attribute() -> None:
    """Test that deprecated properties have `__deprecated__` attribute (PEP 702).

    Note: When using @property over @deprecated (which is what works in practice),
    the `__deprecated__` attribute is set on the property's underlying `fget` function.
    """

    class MyClass:
        @property
        @deprecated(since="2.0.0", removal="3.0.0", alternative="new_property")
        def old_property(self) -> str:
            """Original doc."""
            return "old"

    prop = MyClass.__dict__["old_property"]
    # The __deprecated__ attribute is on the underlying fget function
    assert hasattr(prop.fget, "__deprecated__")
    assert prop.fget.__deprecated__ == "Use new_property instead."
