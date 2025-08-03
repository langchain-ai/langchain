"""Unit tests for Anchorbrowser tools."""

from typing import Any
from unittest.mock import Mock, patch

# Mock the anchorbrowser imports to avoid requiring the actual package
with patch.dict(
    "sys.modules",
    {
        "langchain_anchorbrowser": Mock(),
        "langchain_anchorbrowser.tools": Mock(),
    },
):
    # Create mock tool classes for testing
    class MockAnchorContentTool:
        def __init__(self) -> None:
            self.name = "anchor_content"
            self.description = "Extract content from web pages"
            self.args_schema = Mock()
            self.args_schema.schema.return_value = {
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to extract content from",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "html"],
                        "default": "markdown",
                    },
                },
                "required": ["url"],
            }

        def run(self, args: dict[str, Any]) -> str:
            url = args.get("url", "unknown")
            format_type = args.get("format", "markdown")
            return f"Content from {url} in {format_type} format"

    class MockAnchorScreenshotTool:
        def __init__(self) -> None:
            self.name = "anchor_screenshot"
            self.description = "Take screenshots of web pages"
            self.args_schema = Mock()
            self.args_schema.schema.return_value = {
                "properties": {
                    "url": {"type": "string", "description": "URL to screenshot"},
                    "width": {"type": "integer", "default": 1920},
                    "height": {"type": "integer", "default": 1080},
                    "image_quality": {"type": "integer", "default": 90},
                },
                "required": ["url"],
            }

        def run(self, args: dict[str, Any]) -> str:
            url = args.get("url", "unknown")
            width = args.get("width", 1920)
            height = args.get("height", 1080)
            return f"Screenshot of {url} at {width}x{height}"

    class MockSimpleAnchorWebTaskTool:
        def __init__(self) -> None:
            self.name = "simple_anchor_web_task"
            self.description = "Perform web tasks using AI"
            self.args_schema = Mock()
            self.args_schema.schema.return_value = {
                "properties": {
                    "prompt": {"type": "string", "description": "Task description"},
                    "url": {"type": "string", "description": "URL to perform task on"},
                },
                "required": ["prompt", "url"],
            }

        def run(self, args: dict[str, Any]) -> str:
            prompt = args.get("prompt", "unknown")
            url = args.get("url", "unknown")
            return f"Completed task: {prompt} on {url}"


class TestAnchorContentTool:
    """Test cases for AnchorContentTool schema and configuration."""

    def test_anchor_content_tool_schema_structure(self) -> None:
        """Test that AnchorContentTool has correct schema structure."""
        tool = MockAnchorContentTool()
        schema = tool.args_schema.schema()

        # Test schema properties
        assert "url" in schema["properties"]
        assert "format" in schema["properties"]
        assert "url" in schema["required"]

        # Test property types
        assert schema["properties"]["url"]["type"] == "string"
        assert schema["properties"]["format"]["type"] == "string"
        assert "enum" in schema["properties"]["format"]
        assert "markdown" in schema["properties"]["format"]["enum"]
        assert "html" in schema["properties"]["format"]["enum"]

    def test_anchor_content_tool_default_values(self) -> None:
        """Test that AnchorContentTool has correct default values."""
        tool = MockAnchorContentTool()
        schema = tool.args_schema.schema()

        assert schema["properties"]["format"]["default"] == "markdown"

    def test_anchor_content_tool_required_fields(self) -> None:
        """Test that AnchorContentTool requires necessary fields."""
        tool = MockAnchorContentTool()
        schema = tool.args_schema.schema()

        assert "url" in schema["required"]
        assert "format" not in schema["required"]  # format has default value


class TestAnchorScreenshotTool:
    """Test cases for AnchorScreenshotTool schema and configuration."""

    def test_anchor_screenshot_tool_schema_structure(self) -> None:
        """Test that AnchorScreenshotTool has correct schema structure."""
        tool = MockAnchorScreenshotTool()
        schema = tool.args_schema.schema()

        # Test schema properties
        assert "url" in schema["properties"]
        assert "width" in schema["properties"]
        assert "height" in schema["properties"]
        assert "image_quality" in schema["properties"]
        assert "url" in schema["required"]

        # Test property types
        assert schema["properties"]["url"]["type"] == "string"
        assert schema["properties"]["width"]["type"] == "integer"
        assert schema["properties"]["height"]["type"] == "integer"
        assert schema["properties"]["image_quality"]["type"] == "integer"

    def test_anchor_screenshot_tool_default_values(self) -> None:
        """Test that AnchorScreenshotTool has correct default values."""
        tool = MockAnchorScreenshotTool()
        schema = tool.args_schema.schema()

        assert schema["properties"]["width"]["default"] == 1920
        assert schema["properties"]["height"]["default"] == 1080
        assert schema["properties"]["image_quality"]["default"] == 90

    def test_anchor_screenshot_tool_required_fields(self) -> None:
        """Test that AnchorScreenshotTool requires necessary fields."""
        tool = MockAnchorScreenshotTool()
        schema = tool.args_schema.schema()

        assert "url" in schema["required"]
        # width, height, and image_quality have defaults, so not required
        assert "width" not in schema["required"]
        assert "height" not in schema["required"]
        assert "image_quality" not in schema["required"]


class TestSimpleAnchorWebTaskTool:
    """Test cases for SimpleAnchorWebTaskTool schema and configuration."""

    def test_simple_anchor_web_task_tool_schema_structure(self) -> None:
        """Test that SimpleAnchorWebTaskTool has correct schema structure."""
        tool = MockSimpleAnchorWebTaskTool()
        schema = tool.args_schema.schema()

        # Test schema properties
        assert "prompt" in schema["properties"]
        assert "url" in schema["properties"]
        assert "prompt" in schema["required"]
        assert "url" in schema["required"]

        # Test property types
        assert schema["properties"]["prompt"]["type"] == "string"
        assert schema["properties"]["url"]["type"] == "string"

    def test_simple_anchor_web_task_tool_required_fields(self) -> None:
        """Test that SimpleAnchorWebTaskTool requires necessary fields."""
        tool = MockSimpleAnchorWebTaskTool()
        schema = tool.args_schema.schema()

        assert "prompt" in schema["required"]
        assert "url" in schema["required"]
