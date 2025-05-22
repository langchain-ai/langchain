from langchain_core.documents import (
    Document,  # type: ignore[import-not-found, import-not-found]
)

from langchain_exa import ExaSearchRetriever


def test_exa_retriever() -> None:
    retriever = ExaSearchRetriever()
    res = retriever.invoke("best time to visit japan")
    print(res)  # noqa: T201
    assert len(res) == 10  # default k
    assert isinstance(res, list)
    assert isinstance(res[0], Document)


def test_exa_retriever_highlights() -> None:
    retriever = ExaSearchRetriever(highlights=True)
    res = retriever.invoke("best time to visit japan")
    print(res)  # noqa: T201
    assert isinstance(res, list)
    assert isinstance(res[0], Document)
    highlights = res[0].metadata["highlights"]
    highlight_scores = res[0].metadata["highlight_scores"]
    assert isinstance(highlights, list)
    assert isinstance(highlight_scores, list)
    assert isinstance(highlights[0], str)
    assert isinstance(highlight_scores[0], float)


def test_exa_retriever_advanced_features() -> None:
    retriever = ExaSearchRetriever(
        k=3, text_contents_options={"max_characters": 1000}, summary=True, type="auto"
    )
    res = retriever.invoke("best time to visit japan")
    print(res)  # noqa: T201
    assert len(res) == 3  # requested k=3
    assert isinstance(res, list)
    assert isinstance(res[0], Document)
    # Verify summary is in metadata
    assert "summary" in res[0].metadata
    assert isinstance(res[0].metadata["summary"], str)
    # Verify text was limited
    assert len(res[0].page_content) <= 1000
