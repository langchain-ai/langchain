import os
import unittest
from typing import Any
from unittest.mock import patch

from langchain_community.tools.jina_search.tool import JinaSearch
from langchain_community.utilities.jina_search import JinaSearchAPIWrapper

os.environ["JINA_API_KEY"] = "test_key"


class TestJinaSearchTool(unittest.TestCase):
    @patch(
        "langchain_community.tools.jina_search.tool.JinaSearch.invoke",
        return_value="mocked_result",
    )
    def test_invoke(self, mock_run: Any) -> None:
        query = "Test query text"
        wrapper = JinaSearchAPIWrapper(api_key="test_key")  # type: ignore[arg-type]
        jina_search_tool = JinaSearch(api_wrapper=wrapper)  # type: ignore[call-arg]
        results = jina_search_tool.invoke(query)
        expected_result = "mocked_result"
        assert results == expected_result


if __name__ == "__main__":
    unittest.main()
