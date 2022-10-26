"""Test document functionality."""
from langchain.docstore.document import Document

_PAGE_CONTENT = """This is a page about LangChain.

It is a really cool framework.

What isn't there to love about langchain?

Made in 2022."""


def test_document_summary() -> None:
    """Test that we extract the summary okay."""
    page = Document(page_content=_PAGE_CONTENT)
    assert page.summary == "This is a page about LangChain."


def test_document_lookup() -> None:
    """Test that can lookup things okay."""
    page = Document(page_content=_PAGE_CONTENT)

    # Start with lookup on "LangChain".
    output = page.lookup("LangChain")
    assert output == "(Result 1/2) This is a page about LangChain."

    # Now switch to looking up "framework".
    output = page.lookup("framework")
    assert output == "(Result 1/1) It is a really cool framework."

    # Now switch back to looking up "LangChain", should reset.
    output = page.lookup("LangChain")
    assert output == "(Result 1/2) This is a page about LangChain."

    # Lookup "LangChain" again, should go to the next mention.
    output = page.lookup("LangChain")
    assert output == "(Result 2/2) What isn't there to love about langchain?"


def test_document_lookups_dont_exist() -> None:
    """Test lookup on term that doesn't exist in the document."""
    page = Document(page_content=_PAGE_CONTENT)

    # Start with lookup on "harrison".
    output = page.lookup("harrison")
    assert output == "No Results"


def test_document_lookups_too_many() -> None:
    """Test lookup on term too many times."""
    page = Document(page_content=_PAGE_CONTENT)

    # Start with lookup on "framework".
    output = page.lookup("framework")
    assert output == "(Result 1/1) It is a really cool framework."

    # Now try again, should be exhausted.
    output = page.lookup("framework")
    assert output == "No More Results"
