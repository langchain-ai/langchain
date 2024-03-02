import inspect
import warnings
from typing import Any, Dict

import pytest

from langchain_core._api.deprecation import deprecated, warn_deprecated
from langchain_core.pydantic_v1 import BaseModel


@pytest.mark.parametrize(
    "kwargs, expected_message",
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
def test_warn_deprecated(kwargs: Dict[str, Any], expected_message: str) -> None:
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
    """original doc"""
    return "This is a deprecated function."


@deprecated(since="2.0.0", removal="3.0.0", pending=False)
async def deprecated_async_function() -> str:
    """original doc"""
    return "This is a deprecated async function."


class ClassWithDeprecatedMethods:
    def __init__(self) -> None:
        """original doc"""
        pass

    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_method(self) -> str:
        """original doc"""
        return "This is a deprecated method."

    @deprecated(since="2.0.0", removal="3.0.0")
    async def deprecated_async_method(self) -> str:
        """original doc"""
        return "This is a deprecated async method."

    @classmethod
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_classmethod(cls) -> str:
        """original doc"""
        return "This is a deprecated classmethod."

    @staticmethod
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_staticmethod() -> str:
        """original doc"""
        return "This is a deprecated staticmethod."

    @property
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_property(self) -> str:
        """original doc"""
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
        assert doc.startswith("[*Deprecated*] original doc")

    assert not inspect.iscoroutinefunction(deprecated_function)


@pytest.mark.asyncio
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
        assert doc.startswith("[*Deprecated*] original doc")

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
            "The function `deprecated_method` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )

        doc = obj.deprecated_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("[*Deprecated*] original doc")

    assert not inspect.iscoroutinefunction(obj.deprecated_method)


@pytest.mark.asyncio
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
            "The function `deprecated_async_method` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )

        doc = obj.deprecated_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("[*Deprecated*] original doc")

    assert inspect.iscoroutinefunction(obj.deprecated_async_method)


def test_deprecated_classmethod() -> None:
    """Test deprecated classmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        ClassWithDeprecatedMethods.deprecated_classmethod()
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The function `deprecated_classmethod` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )

        doc = ClassWithDeprecatedMethods.deprecated_classmethod.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("[*Deprecated*] original doc")


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
            "The function `deprecated_staticmethod` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )
        doc = ClassWithDeprecatedMethods.deprecated_staticmethod.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("[*Deprecated*] original doc")


def test_deprecated_property() -> None:
    """Test deprecated staticmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = ClassWithDeprecatedMethods()
        assert obj.deprecated_property == "This is a deprecated property."

        assert len(warning_list) == 1
        warning = warning_list[0].message

        assert str(warning) == (
            "The function `deprecated_property` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )
        doc = ClassWithDeprecatedMethods.deprecated_property.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("[*Deprecated*] original doc")


def test_whole_class_deprecation() -> None:
    """Test whole class deprecation."""

    # Test whole class deprecation
    @deprecated(since="2.0.0", removal="3.0.0")
    class DeprecatedClass:
        def __init__(self) -> None:
            """original doc"""
            pass

        @deprecated(since="2.0.0", removal="3.0.0")
        def deprecated_method(self) -> str:
            """original doc"""
            return "This is a deprecated method."

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = DeprecatedClass()
        assert obj.deprecated_method() == "This is a deprecated method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class `tests.unit_tests._api.test_deprecation.DeprecatedClass` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The function `deprecated_method` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )
        # [*Deprecated*] should be inserted only once:
        if obj.__doc__ is not None:
            assert obj.__doc__.count("[*Deprecated*]") == 1


def test_whole_class_inherited_deprecation() -> None:
    """Test whole class deprecation for inherited class.

    The original version of deprecation decorator created duplicates with
    '[*Deprecated*]'.
    """

    # Test whole class deprecation
    @deprecated(since="2.0.0", removal="3.0.0")
    class DeprecatedClass:
        def __init__(self) -> None:
            """original doc"""
            pass

        @deprecated(since="2.0.0", removal="3.0.0")
        def deprecated_method(self) -> str:
            """original doc"""
            return "This is a deprecated method."

    @deprecated(since="2.2.0", removal="3.2.0")
    class InheritedDeprecatedClass(DeprecatedClass):
        """Inherited deprecated class."""

        def __init__(self) -> None:
            """original doc"""
            pass

        @deprecated(since="2.2.0", removal="3.2.0")
        def deprecated_method(self) -> str:
            """original doc"""
            return "This is a deprecated method."

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = DeprecatedClass()
        assert obj.deprecated_method() == "This is a deprecated method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class `tests.unit_tests._api.test_deprecation.DeprecatedClass` was "
            "deprecated in tests 2.0.0 and will be removed in 3.0.0"
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The function `deprecated_method` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )
        # if [*Deprecated*] was inserted only once:
        if obj.__doc__ is not None:
            assert obj.__doc__.count("[*Deprecated*]") == 1

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = InheritedDeprecatedClass()
        assert obj.deprecated_method() == "This is a deprecated method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class "
            "`tests.unit_tests._api.test_deprecation.InheritedDeprecatedClass` "
            "was deprecated in tests 2.2.0 and will be removed in 3.2.0"
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The function `deprecated_method` was deprecated in "
            "LangChain 2.2.0 and will be removed in 3.2.0"
        )
        # if [*Deprecated*] was inserted only once:
        if obj.__doc__ is not None:
            assert obj.__doc__.count("[*Deprecated*]") == 1
            assert "[*Deprecated*] Inherited deprecated class." in obj.__doc__


# Tests with pydantic models
class MyModel(BaseModel):
    @deprecated(since="2.0.0", removal="3.0.0")
    def deprecated_method(self) -> str:
        """original doc"""
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
            "The function `deprecated_method` was deprecated in "
            "LangChain 2.0.0 and will be removed in 3.0.0"
        )

        doc = obj.deprecated_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith("[*Deprecated*] original doc")
