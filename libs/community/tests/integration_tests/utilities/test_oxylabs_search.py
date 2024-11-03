"""Integration test for OxylabsSearchAPIWrapper."""

import pytest

from langchain_community.utilities import OxylabsSearchAPIWrapper


def oxylabs_installed() -> bool:
    try:
        from oxylabs import RealtimeClient  # noqa: F401

        return True

    except Exception:
        return False


@pytest.mark.skipif(
    not oxylabs_installed(), reason="requires oxylabs package - `pip install oxylabs`"
)
def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = OxylabsSearchAPIWrapper()  # type: ignore[call-arg]
    output = chain.run("Python programming language")
    assert "high-level, general-purpose programming language" in output
    assert "extensions" in output
    assert ".py" in output
    assert "Guido van Rossum" in output


@pytest.mark.skipif(
    not oxylabs_installed(), reason="requires oxylabs package - `pip install oxylabs`"
)
async def test_async_call() -> None:
    """Test that call gives the correct answer."""
    chain = OxylabsSearchAPIWrapper()  # type: ignore[call-arg]
    output = chain.run("Python programming language")
    assert "high-level, general-purpose programming language" in output
    assert "extensions" in output
    assert ".py" in output
    assert "Guido van Rossum" in output
