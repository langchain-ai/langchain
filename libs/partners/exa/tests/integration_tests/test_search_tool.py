from langchain_exa import (
    ExaSearchResults,  # type: ignore[import-not-found, import-not-found]
)


def test_search_tool() -> None:
    tool = ExaSearchResults()
    res = tool.invoke({"query": "best time to visit japan", "num_results": 5})
    print(res)  # noqa: T201
    assert not isinstance(res, str)  # str means error for this tool\
