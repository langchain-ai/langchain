"""Test the public API of the tools package."""
from langchain.tools import __all__ as public_api

_EXPECTED = [
    "AIPluginTool",
    "APIOperation",
    "BaseTool",
    "BaseTool",
    "BaseTool",
    "BingSearchResults",
    "BingSearchRun",
    "ClickTool",
    "CopyFileTool",
    "CurrentWebPageTool",
    "DeleteFileTool",
    "DuckDuckGoSearchResults",
    "DuckDuckGoSearchRun",
    "ExtractHyperlinksTool",
    "ExtractTextTool",
    "FileSearchTool",
    "GetElementsTool",
    "GooglePlacesTool",
    "GoogleSearchResults",
    "GoogleSearchRun",
    "GoogleSerperResults",
    "GoogleSerperRun",
    "HumanInputRun",
    "IFTTTWebhook",
    "ListDirectoryTool",
    "MoveFileTool",
    "NavigateBackTool",
    "NavigateTool",
    "OpenAPISpec",
    "ReadFileTool",
    "SceneXplainTool",
    "ShellTool",
    "StructuredTool",
    "Tool",
    "VectorStoreQATool",
    "VectorStoreQAWithSourcesTool",
    "WikipediaQueryRun",
    "WolframAlphaQueryRun",
    "WriteFileTool",
    "ZapierNLAListActions",
    "ZapierNLARunAction",
    "tool",
]


def test_public_api() -> None:
    """Test for regressions or changes in the public API."""
    # Check that the public API is as expected
    assert public_api == sorted(_EXPECTED)
