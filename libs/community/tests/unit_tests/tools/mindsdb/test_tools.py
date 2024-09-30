from unittest.mock import MagicMock

from langchain_community.tools import AIMindTool


def test_ai_mind_tool() -> None:
    mock_api_wrapper = MagicMock()
    mock_api_wrapper.run.return_value = "dummy response"

    tool = AIMindTool.construct(api_wrapper=mock_api_wrapper)

    query = "dummy query"

    result = tool.run(query)
    assert result == "dummy response"
