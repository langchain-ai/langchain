"""Test Minimax API wrapper."""
from langchain.llms.minimax import Minimax


def test_minimax_call() -> None:
    """Test valid call to minimax."""
    llm = Minimax(max_tokens=10)
    output = llm("Hello world!")
    assert isinstance(output, str)
