"""Pytest configuration for langchain-sarvam tests."""

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Mark tests that require API keys."""
    for item in items:
        if "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.scheduled)


@pytest.fixture(scope="session")
def sarvam_api_key() -> str:
    """Get Sarvam API key from environment."""
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        pytest.skip("SARVAM_API_KEY not set")
    return key
