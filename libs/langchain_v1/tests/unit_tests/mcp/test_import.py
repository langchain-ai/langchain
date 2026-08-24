import pytest
from mcp import StdioServerParameters
from mcp.client.session_group import SseServerParameters, StreamableHttpParameters

from langchain.mcp.sessions import _parse_mapping


def test_import() -> None:
    """Test that the code can be imported"""
    from langchain.mcp import (  # noqa: F401, PLC0415
        callbacks,
        client,
        prompts,
        resources,
        tools,
    )


def test_connection_mapping_parses_into_sdk_parameters() -> None:
    """Mapping configs are validated by the SDK's models, not types declared here."""
    params, options = _parse_mapping(
        {"transport": "stdio", "command": "python3", "args": ["s.py"], "env": {"A": "b"}}
    )
    assert isinstance(params, StdioServerParameters)
    assert (params.command, params.args, params.env) == ("python3", ["s.py"], {"A": "b"})
    assert options == {}

    # Transport is inferred when omitted, and client options ride alongside.
    params, options = _parse_mapping({"url": "https://x/mcp", "mode": "legacy"})
    assert isinstance(params, StreamableHttpParameters)
    assert params.url == "https://x/mcp"
    assert options == {"mode": "legacy"}

    params, _ = _parse_mapping({"transport": "sse", "url": "https://x/sse"})
    assert isinstance(params, SseServerParameters)


def test_unknown_transport_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported transport"):
        _parse_mapping({"transport": "carrier-pigeon", "url": "https://x"})
