"""Test Playwright's Tools."""

from unittest.mock import Mock

import pytest

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit


@pytest.mark.requires("playwright")
@pytest.mark.requires("bs4")
def test_playwright_tools_schemas() -> None:
    """Test calling 'tool_call_schema' for every tool to check to init issues."""

    from playwright.sync_api import Browser

    sync_browser = Mock(spec=Browser)
    tools = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser).get_tools()

    for tool in tools:
        try:
            tool.tool_call_schema
        except Exception as e:
            raise AssertionError(
                f"Error for '{tool.name}' tool: {type(e).__name__}: {e}"
            ) from e
