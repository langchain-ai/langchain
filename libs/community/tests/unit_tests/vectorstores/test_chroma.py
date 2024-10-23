from langchain_core.documents import Document

from langchain_community.vectorstores.chroma import _results_to_docs_and_scores

_PAGE_CONTENT = """This is a page about LangChain.

It is a really cool framework.

What isn't there to love about langchain?

Made in 2022."""


def test_results_to_docs_and_scores():
    """Test the results for correct document information."""
    input_data = {
        "ids": [["1"]],
        "embeddings": None,
        "documents": [[_PAGE_CONTENT]],
        "metadatas": [[{"source": "1"}]],
        "distances": [[0.1111]],
    }

    results = _results_to_docs_and_scores(input_data)

    expected_results = (
        Document(id="1", page_content=_PAGE_CONTENT, metadata={"source": "1"}),
        0.1111,
    )

    assert results[0] == expected_results
