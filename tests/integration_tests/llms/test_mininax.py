"""Test Minimax API wrapper.
In order to run this test, you need to have an account on Minimax.
"""

from langchain.llms.minimax import MiniMaxCompletion


def test_minimax_call() -> None:
    """Test valid call to Minimax."""
    llm = MiniMaxCompletion()
    output = llm(
        "A chain is a serial assembly of connected pieces, called links, \
        typically made of metal, with an overall character similar to that\
        of a rope in that it is flexible and curved in compression but \
        linear, rigid, and load-bearing in tension. A chain may consist\
        of two or more links."
    )
    assert isinstance(output, str)
