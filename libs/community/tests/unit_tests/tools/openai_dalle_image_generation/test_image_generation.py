from unittest.mock import MagicMock

from langchain_community.tools.openai_dalle_image_generation import (
    OpenAIDALLEImageGenerationTool,
)


def test_generate_image() -> None:
    """Test OpenAI DALLE Image Generation."""
    mock_api_resource = MagicMock()
    # bypass pydantic validation as openai is not a package dependency
    tool = OpenAIDALLEImageGenerationTool.construct(api_wrapper=mock_api_resource)
    tool_input = {"query": "parrot on a branch"}
    result = tool.run(tool_input)
    assert result.startswith("https://")
