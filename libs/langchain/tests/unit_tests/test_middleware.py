import pytest
from langchain_core.messages import ToolMessage
from langchain_classic.agents.tool_calling_agent import HumanInTheLoopMiddleware

def test_middleware_approval_granted():
    """Test when human approves the tool message."""
    def dummy_approval(msg: ToolMessage) -> bool:
        return True

    middleware = HumanInTheLoopMiddleware(approval_callback=dummy_approval)
    msg = ToolMessage(content="Success data", tool_call_id="123")

    processed = middleware.process_tool_message(msg)
    assert processed.content == "Success data"
    assert processed.tool_call_id == "123"

def test_middleware_approval_rejected():
    """Test when human rejects the tool message."""
    def dummy_approval(msg: ToolMessage) -> bool:
        return False

    middleware = HumanInTheLoopMiddleware(approval_callback=dummy_approval)
    msg = ToolMessage(content="Sensitive data", tool_call_id="456")

    processed = middleware.process_tool_message(msg)
    assert "rejected" in processed.content
    assert processed.status == "error"

def test_middleware_modification():
    """Test modifying the tool message content."""
    def dummy_modifier(msg: ToolMessage) -> ToolMessage:
        msg.content = "Modified content"
        return msg

    middleware = HumanInTheLoopMiddleware(modifier_callback=dummy_modifier)
    msg = ToolMessage(content="Original content", tool_call_id="789")

    processed = middleware.process_tool_message(msg)
    assert processed.content == "Modified content"
