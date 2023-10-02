import pytest as pytest

from langchain.utilities import ArxivAPIWrapper


@pytest.mark.requires("arxiv")
def test_is_arxiv_identifier() -> None:
    """Test that is_arxiv_identifier returns True for valid arxiv identifiers"""
    api_client = ArxivAPIWrapper()
    assert api_client.is_arxiv_identifier("1605.08386v1")
    assert api_client.is_arxiv_identifier("0705.0123")
    assert api_client.is_arxiv_identifier("2308.07912")
    assert api_client.is_arxiv_identifier("9603067 2308.07912 2308.07912")
    assert not api_client.is_arxiv_identifier("12345")
    assert not api_client.is_arxiv_identifier("0705.012")
    assert not api_client.is_arxiv_identifier("0705.012300")
    assert not api_client.is_arxiv_identifier("1605.08386w1")
