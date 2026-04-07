from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool, ToolException


class FakeTool(BaseTool):
    name: str = "fake_tool"
    description: str = "A fake tool for testing"

    def _run(self, query: str) -> str:
        return f"result for {query}"


def test_toolkit_wraps_tools() -> None:
    """AIIdentityToolkit.get_tools returns wrapped tool list."""
    with patch(
        "langchain_ai_identity.tools.enforce_access",
        return_value={"decision": "allow"},
    ):
        from langchain_ai_identity import AIIdentityToolkit

        toolkit = AIIdentityToolkit(
            tools=[FakeTool()],
            agent_id="test-uuid",
            api_key="aid_sk_test",
        )
        tools = toolkit.get_tools()
        assert len(tools) == 1


def test_wrapped_tool_enforces() -> None:
    """Wrapped tools call enforce_access before execution."""
    with patch(
        "langchain_ai_identity.tools.enforce_access",
        return_value={"decision": "allow"},
    ) as mock_enforce:
        from langchain_ai_identity import AIIdentityToolkit

        toolkit = AIIdentityToolkit(
            tools=[FakeTool()],
            agent_id="test-uuid",
            api_key="aid_sk_test",
        )
        tools = toolkit.get_tools()
        result = tools[0]._run("test")
        mock_enforce.assert_called_once()
        assert result == "result for test"


def test_wrapped_tool_deny_raises() -> None:
    """Wrapped tools raise ToolException when denied."""
    with patch(
        "langchain_ai_identity.tools.enforce_access",
        return_value={"decision": "deny", "reason": "policy violation"},
    ):
        from langchain_ai_identity import AIIdentityToolkit

        toolkit = AIIdentityToolkit(
            tools=[FakeTool()],
            agent_id="test-uuid",
            api_key="aid_sk_test",
        )
        tools = toolkit.get_tools()
        with pytest.raises(ToolException, match="policy violation"):
            tools[0]._run("test")
