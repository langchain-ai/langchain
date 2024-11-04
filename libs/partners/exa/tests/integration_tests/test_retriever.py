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
