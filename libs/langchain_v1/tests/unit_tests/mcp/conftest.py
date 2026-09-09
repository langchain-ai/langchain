"""Test configuration for the MCP adapter tests."""

import jsonschema  # type: ignore[import-untyped]  # noqa: F401  # side effect, see below
import pytest
from blockbuster import BlockBuster
from pydantic import VERSION as PYDANTIC_VERSION

_MCP_MIN_PYDANTIC = (2, 12)
"""`mcp` and `mcp-types` require pydantic 2.12, well above this package's floor."""

if tuple(int(part) for part in PYDANTIC_VERSION.split(".")[:2]) < _MCP_MIN_PYDANTIC:
    # The `mcp` extra cannot install on an older pydantic, but the pydantic
    # compatibility matrix pins one *after* resolution and then runs the whole
    # unit suite. Skip this directory there rather than fail every MCP test on
    # a `model_validate(..., by_name=...)` the older pydantic has no parameter
    # for. Nothing outside the `mcp` extra is affected.
    collect_ignore_glob = ["*.py"]

# `mcp.client.session.validate_tool_result` imports `jsonschema` lazily, and
# `jsonschema_specifications` reads its bundled schemas from disk on first
# import. Importing it here means that one-time filesystem work happens before
# any test enters an event loop, so `blockbuster` does not flag it as a blocking
# call inside the first MCP tool call to run.

_FASTMCP_DEPENDENCY_PROBE = ("fastmcp/server/dependencies.py", "is_docket_available")


@pytest.fixture(autouse=True)
def _allow_fastmcp_dependency_probe(blockbuster: BlockBuster) -> None:
    """Let FastMCP's own optional-dependency probe touch the filesystem.

    Entering a FastMCP server `Context` — which happens on every request an
    in-process MCP server handles — calls `is_docket_available()`, resolving an
    installed version through `importlib.metadata`. That walks the filesystem on
    the event loop from inside FastMCP, where the adapter cannot intervene.
    Without this allowance every test that drives an in-memory MCP server fails
    inside FastMCP instead of in the code under test.

    Scoped to that one call site, so blocking calls made anywhere else still
    fail the suite.
    """
    for function in blockbuster.functions.values():
        function.can_block_in(*_FASTMCP_DEPENDENCY_PROBE)
