import inspect
import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from langchain_core._api.beta_decorator import beta, warn_beta


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {
                "name": "OldClass",
                "obj_type": "class",
            },
            "The class `OldClass` is in beta. It is actively being worked on, so the "
            "API may change.",
        ),
        (
            {
                "message": "This is a custom message",
                "name": "FunctionA",
                "obj_type": "",
                "addendum": "",
            },
            "This is a custom message",
        ),
        (
            {
                "message": "",
                "name": "SomeFunction",
                "obj_type": "",
                "addendum": "Please migrate your code.",
            },
            "`SomeFunction` is in beta. It is actively being worked on, "
            "so the API may "
            "change. Please migrate your code.",
        ),
    ],
)
def test_warn_beta(kwargs: dict[str, Any], expected_message: str) -> None:
    """Test warn beta."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        warn_beta(**kwargs)

        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == expected_message


@beta()
def beta_function() -> str:
    """Original doc."""
    return "This is a beta function."


@beta()
async def beta_async_function() -> str:
    """Original doc."""
    return "This is a beta async function."


class ClassWithBetaMethods:
    def __init__(self) -> None:
        """Original doc."""

    @beta()
    def beta_method(self) -> str:
        """Original doc."""
        return "This is a beta method."

    @beta()
    async def beta_async_method(self) -> str:
        """Original doc."""
        return "This is a beta async method."

    @classmethod
    @beta()
    def beta_classmethod(cls) -> str:
        """Original doc."""
        return "This is a beta classmethod."

    @staticmethod
    @beta()
    def beta_staticmethod() -> str:
        """Original doc."""
        return "This is a beta staticmethod."

    @property
    @beta()
    def beta_property(self) -> str:
        """Original doc."""
        return "This is a beta property."


def test_beta_function() -> None:
    """Test beta function."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert beta_function() == "This is a beta function."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The function `beta_function` is in beta. It is actively being "
            "worked on, "
            "so the API may change."
        )

        doc = beta_function.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")

    assert not inspect.iscoroutinefunction(beta_function)


async def test_beta_async_function() -> None:
    """Test beta async function."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert await beta_async_function() == "This is a beta async function."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The function `beta_async_function` is in beta. "
            "It is actively being worked on, so the API may change."
        )

        doc = beta_function.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")

    assert inspect.iscoroutinefunction(beta_async_function)


def test_beta_method() -> None:
    """Test beta method."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        obj = ClassWithBetaMethods()
        assert obj.beta_method() == "This is a beta method."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `ClassWithBetaMethods.beta_method` is in beta. It is actively "
            "being worked on, so "
            "the API may change."
        )

        doc = obj.beta_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")

    assert not inspect.iscoroutinefunction(obj.beta_method)


async def test_beta_async_method() -> None:
    """Test beta method."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        obj = ClassWithBetaMethods()
        assert await obj.beta_async_method() == "This is a beta async method."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `ClassWithBetaMethods.beta_async_method` is in beta. "
            "It is actively being worked on, so the API may change."
        )

        doc = obj.beta_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")

    assert inspect.iscoroutinefunction(obj.beta_async_method)


def test_beta_classmethod() -> None:
    """Test beta classmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        ClassWithBetaMethods.beta_classmethod()
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `ClassWithBetaMethods.beta_classmethod` is in beta. "
            "It is actively being worked on, so the API may change."
        )

        doc = ClassWithBetaMethods.beta_classmethod.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")


def test_beta_staticmethod() -> None:
    """Test beta staticmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert (
            ClassWithBetaMethods.beta_staticmethod() == "This is a beta staticmethod."
        )
        assert len(warning_list) == 1
        warning = warning_list[0].message

        assert str(warning) == (
            "The method `ClassWithBetaMethods.beta_staticmethod` is in beta. "
            "It is actively being worked on, so the API may change."
        )
        doc = ClassWithBetaMethods.beta_staticmethod.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")


def test_beta_property() -> None:
    """Test beta staticmethod."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = ClassWithBetaMethods()
        assert obj.beta_property == "This is a beta property."

        assert len(warning_list) == 1
        warning = warning_list[0].message

        assert str(warning) == (
            "The method `ClassWithBetaMethods.beta_property` is in beta. "
            "It is actively being worked on, so the API may change."
        )
        doc = ClassWithBetaMethods.beta_property.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")


def test_whole_class_beta() -> None:
    """Test whole class beta status."""

    @beta()
    class BetaClass:
        def __init__(self) -> None:
            """Original doc."""

        @beta()
        def beta_method(self) -> str:
            """Original doc."""
            return "This is a beta method."

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = BetaClass()
        assert obj.beta_method() == "This is a beta method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class `test_whole_class_beta.<locals>.BetaClass` is in beta. "
            "It is actively being worked on, so the "
            "API may change."
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The method `test_whole_class_beta.<locals>.BetaClass.beta_method` "
            "is in beta. It is actively being worked on, so "
            "the API may change."
        )


def test_whole_class_inherited_beta() -> None:
    """Test whole class beta status for inherited class.

    The original version of beta decorator created duplicates with
    '.. beta::'.
    """

    # Test whole class beta status
    @beta()
    class BetaClass:
        @beta()
        def beta_method(self) -> str:
            """Original doc."""
            return "This is a beta method."

    @beta()
    class InheritedBetaClass(BetaClass):
        @beta()
        def beta_method(self) -> str:
            """Original doc."""
            return "This is a beta method 2."

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = BetaClass()
        assert obj.beta_method() == "This is a beta method."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class `test_whole_class_inherited_beta.<locals>.BetaClass` "
            "is in beta. It is actively being worked on, so the "
            "API may change."
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The method `test_whole_class_inherited_beta.<locals>.BetaClass."
            "beta_method` is in beta. It is actively being worked on, so "
            "the API may change."
        )

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        obj = InheritedBetaClass()
        assert obj.beta_method() == "This is a beta method 2."

        assert len(warning_list) == 2
        warning = warning_list[0].message
        assert str(warning) == (
            "The class `test_whole_class_inherited_beta.<locals>.InheritedBetaClass` "
            "is in beta. "
            "It is actively being worked on, so the "
            "API may change."
        )

        warning = warning_list[1].message
        assert str(warning) == (
            "The method `test_whole_class_inherited_beta.<locals>.InheritedBetaClass."
            "beta_method` is in beta. "
            "It is actively being worked on, so "
            "the API may change."
        )

        # if .. beta:: was inserted only once:
        if obj.__doc__ is not None:
            assert obj.__doc__.count(".. beta::") == 1


# Tests with pydantic models
class MyModel(BaseModel):
    @beta()
    def beta_method(self) -> str:
        """Original doc."""
        return "This is a beta method."


def test_beta_method_pydantic() -> None:
    """Test beta method."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        obj = MyModel()
        assert obj.beta_method() == "This is a beta method."
        assert len(warning_list) == 1
        warning = warning_list[0].message
        assert str(warning) == (
            "The method `MyModel.beta_method` is in beta. It is actively being "
            "worked on, so "
            "the API may change."
        )

        doc = obj.beta_method.__doc__
        assert isinstance(doc, str)
        assert doc.startswith(".. beta::")
