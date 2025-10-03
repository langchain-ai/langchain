"""Standard tests."""


class BaseStandardTests:
    """Base class for standard tests."""

    def test_no_overrides_DO_NOT_OVERRIDE(self) -> None:  # noqa: N802
        """Test that no standard tests are overridden."""
        # Find path to standard test implementations
        comparison_class = None

        def explore_bases(cls: type) -> None:
            nonlocal comparison_class
            for base in cls.__bases__:
                if base.__module__.startswith("langchain_tests."):
                    if comparison_class is None:
                        comparison_class = base
                    else:
                        msg = (
                            "Multiple standard test base classes found: "
                            f"{comparison_class}, {base}"
                        )
                        raise ValueError(msg)
                else:
                    explore_bases(base)

        explore_bases(self.__class__)
        assert comparison_class is not None, "No standard test base class found."

        print(f"Comparing {self.__class__} to {comparison_class}")  # noqa: T201

        running_tests = {method for method in dir(self) if method.startswith("test_")}
        base_tests = {
            method for method in dir(comparison_class) if method.startswith("test_")
        }
        deleted_tests = base_tests - running_tests
        assert not deleted_tests, f"Standard tests deleted: {deleted_tests}"

        overridden_tests = [
            method
            for method in base_tests
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
