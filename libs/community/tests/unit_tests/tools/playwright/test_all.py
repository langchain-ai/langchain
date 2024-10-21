"""Test Playwright's Tools."""

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser


def test_playwright_tools_schemas() -> None:
    """Test calling 'tool_call_schema' for every tool to check to init issues."""

    sync_browser = create_sync_playwright_browser()
    tools = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser).get_tools()

    for tool in tools:
        try:
            tool.tool_call_schema
        except Exception as e:
            raise AssertionError(
                f"Error for '{tool.name}' tool: {type(e).__name__}: {e}"
            ) from e
