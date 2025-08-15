import pytest  # type: ignore[import-not-found]


@pytest.mark.compile
def test_placeholder() -> None:
    """Used for compiling integration tests without running any real tests."""
    pass
