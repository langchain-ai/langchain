from abc import ABC
from typing import Type


class BaseStandardTests(ABC):
    """
    :private:
    """

    def test_no_overrides_DO_NOT_OVERRIDE(self) -> None:
        """
        Test that no standard tests are overridden.

        :private:
        """
        # find path to standard test implementations
        comparison_class = None

        def explore_bases(cls: Type) -> None:
            nonlocal comparison_class
            for base in cls.__bases__:
                if base.__module__.startswith("langchain_tests."):
                    if comparison_class is None:
                        comparison_class = base
                    else:
                        raise ValueError(
                            "Multiple standard test base classes found: "
                            f"{comparison_class}, {base}"
                        )
                else:
                    explore_bases(base)

        explore_bases(self.__class__)
        assert comparison_class is not None, "No standard test base class found."

        print(f"Comparing {self.__class__} to {comparison_class}")  # noqa: T201

        running_tests = set(
            [method for method in dir(self) if method.startswith("test_")]
        )
        base_tests = set(
            [method for method in dir(comparison_class) if method.startswith("test_")]
        )
        non_standard_tests = running_tests - base_tests
        assert not non_standard_tests, f"Non-standard tests found: {non_standard_tests}"
        deleted_tests = base_tests - running_tests
        assert not deleted_tests, f"Standard tests deleted: {deleted_tests}"

        overridden_tests = [
            method
            for method in running_tests
            if getattr(self.__class__, method) is not getattr(comparison_class, method)
        ]

        def is_xfail(method: str) -> bool:
            m = getattr(self.__class__, method)
            if not hasattr(m, "pytestmark"):
                return False
            marks = m.pytestmark
            return any(
                mark.name == "xfail" and mark.kwargs.get("reason") for mark in marks
            )

        overridden_not_xfail = [
            method for method in overridden_tests if not is_xfail(method)
        ]
        assert not overridden_not_xfail, (
            "Standard tests overridden without "
            f'@pytest.mark.xfail(reason="..."): {overridden_not_xfail}\n'
            "Note: reason is required to explain why the standard test has an expected "
            "failure."
        )
