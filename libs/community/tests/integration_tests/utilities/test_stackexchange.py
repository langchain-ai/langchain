"""Integration test for Stack Exchange."""

from langchain_community.utilities import StackExchangeAPIWrapper


def test_call() -> None:
    """Test that call runs."""
    stackexchange = StackExchangeAPIWrapper()  # type: ignore[call-arg]
    output = stackexchange.run("zsh: command not found: python")
    assert output != "hello"


def test_failure() -> None:
    """Test that call that doesn't run."""
    stackexchange = StackExchangeAPIWrapper()  # type: ignore[call-arg]
    output = stackexchange.run("sjefbsmnf")
    assert output == "No relevant results found for 'sjefbsmnf' on Stack Overflow"


def test_success() -> None:
    """Test that call that doesn't run."""
    stackexchange = StackExchangeAPIWrapper()  # type: ignore[call-arg]
    output = stackexchange.run("zsh: command not found: python")
    assert "zsh: command not found: python" in output
