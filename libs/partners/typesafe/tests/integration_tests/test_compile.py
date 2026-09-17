"""Test compilation of integration tests."""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Provide a target for the integration-test compilation job."""
