from langchain_exa import ExaSearchResults


def test_search_tool() -> None:
    tool = ExaSearchResults()
    res = tool.invoke({"query": "best time to visit japan", "num_results": 5})
    print(res)
    assert not isinstance(res, str)  # str means error for this tool\
