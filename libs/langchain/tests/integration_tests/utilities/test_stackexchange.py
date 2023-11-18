"""Integration test for Stack Exchange."""
from langchain.utilities.stackexchange import StackExchangeAPIWrapper


def test_call() -> None:
    """Test that call runs."""
    stackexchange = StackExchangeAPIWrapper()
    output = stackexchange.run("zsh: command not found: python")
    print(output)
    assert output == "hello"