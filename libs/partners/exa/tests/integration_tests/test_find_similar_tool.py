from langchain_exa import (
    ExaFindSimilarResults,  # type: ignore[import-not-found, import-not-found]
)


def test_similarity_tool() -> None:
    tool = ExaFindSimilarResults()
    res = tool.invoke(
        {
            "url": "https://boutiquejapan.com/when-is-the-best-time-of-year-to-visit-japan/",
            "num_results": 5,
        }
    )
    print(res)  # noqa: T201
    assert not isinstance(res, str)  # str means error for this tool
