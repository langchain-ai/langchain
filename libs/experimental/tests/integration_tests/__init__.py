"""All integration tests (tests that call out to an external API)."""
import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Used for compiling integration tests without running any real tests."""
    pass
