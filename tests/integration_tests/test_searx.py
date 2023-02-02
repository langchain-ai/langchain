"""Integration test for Searx API."""
from langchain.searx import SearxAPIWrapper

def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = SearxAPIWrapper()
    output = chain.run("Who is Geralt Of Rivia ?")
