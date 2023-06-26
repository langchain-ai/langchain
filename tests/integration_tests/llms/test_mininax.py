"""Test Clarifai API wrapper.
In order to run this test, you need to have an account on Minimax.
You can sign up for at https://api.minimax.chat/
"""

from langchain.llms.minimax import MiniMaxChatCompletion


def test_minimax_call() -> None:
    """Test valid call to clarifai."""
    llm = MiniMaxChatCompletion()
    output = llm(
        "A chain is a serial assembly of connected pieces, called links, \
        typically made of metal, with an overall character similar to that\
        of a rope in that it is flexible and curved in compression but \
        linear, rigid, and load-bearing in tension. A chain may consist\
        of two or more links."
    )
    assert isinstance(output, str)
