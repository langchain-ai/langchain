"""Unit tests for MCP Discovery Tool."""

from unittest.mock import patch

import requests

from langchain_classic.tools.mcp_discovery.tool import MCPDiscoveryTool


class TestMCPDiscoveryTool:
    """Test cases for MCPDiscoveryTool."""

    def test_discovery_success(self) -> None:
        """Tool formats a valid MCP Discovery response correctly."""
        mock_payload = {
            "recommendations": [
                {
                    "server": "gmail-server",
                    "name": "Gmail MCP Server",
                    "description": "Send and read emails via Gmail",
                    "install_command": "npx -y @modelcontextprotocol/server-gmail",
                    "confidence": 0.87,
                    "category": "communication",
                }
            ],
            "total_found": 1,
            "query_time_ms": 42,
        }

        with patch(
            "langchain_classic.tools.mcp_discovery.tool.requests.post"
        ) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_payload

            tool = MCPDiscoveryTool()
            result = tool.run({"need": "send emails", "limit": 3})

            assert "Gmail MCP Server" in result
            assert "Install command" in result
            assert "Confidence score" in result

    def test_discovery_empty_results(self) -> None:
        """Tool handles empty discovery results gracefully."""
        with patch(
            "langchain_classic.tools.mcp_discovery.tool.requests.post"
        ) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"recommendations": []}

            tool = MCPDiscoveryTool()
            result = tool.run({"need": "unknown capability"})

            assert "No MCP servers found" in result

    def test_discovery_network_error(self) -> None:
        """Tool handles network/API errors safely."""
        with patch(
            "langchain_classic.tools.mcp_discovery.tool.requests.post",
            side_effect=requests.RequestException("API down"),
        ):
            tool = MCPDiscoveryTool()
            result = tool.run({"need": "email"})

            assert "Failed to contact MCP Discovery service" in result
