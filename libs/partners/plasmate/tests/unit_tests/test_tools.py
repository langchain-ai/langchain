"""Unit tests for Plasmate LangChain tools."""

from langchain_plasmate import PlasmateFetchTool, PlasmateNavigateTool, PlasmateLoader


def test_fetch_tool_instantiation() -> None:
    """Test that PlasmateFetchTool can be instantiated."""
    tool = PlasmateFetchTool()
    assert tool.name == "plasmate_fetch"
    assert "web page" in tool.description.lower()


def test_navigate_tool_instantiation() -> None:
    """Test that PlasmateNavigateTool can be instantiated."""
    tool = PlasmateNavigateTool()
    assert tool.name == "plasmate_navigate"
    assert "interactive" in tool.description.lower()


def test_loader_instantiation() -> None:
    """Test that PlasmateLoader can be instantiated."""
    loader = PlasmateLoader(urls=["https://example.com"])
    assert loader.urls == ["https://example.com"]


def test_fetch_tool_schema() -> None:
    """Test that PlasmateFetchTool has correct input schema."""
    tool = PlasmateFetchTool()
    schema = tool.args_schema.model_json_schema()
    assert "url" in schema["properties"]


def test_navigate_tool_schema() -> None:
    """Test that PlasmateNavigateTool has correct input schema."""
    tool = PlasmateNavigateTool()
    schema = tool.args_schema.model_json_schema()
    assert "url" in schema["properties"]
    assert "extract_links" in schema["properties"]
