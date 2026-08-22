from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from langchain.agents.middleware.human_in_the_loop import Action, HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentState


def test_human_in_the_loop_middleware_syncs_additional_kwargs() -> None:
    """Test that HITL middleware syncs additional_kwargs during edits."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "test_tool", "args": {"input": "original"}, "id": "1"}],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "arguments": '{"input": "original"}',
                        "name": "test_tool",
                    },
                    "type": "function",
                }
            ]
        },
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_edit(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="test_tool",
                        args={"input": "edited"},
                    ),
                }
            ]
        }

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        updated_ai_message = result["messages"][0]
        
        # Verify tool_calls is updated
        assert updated_ai_message.tool_calls[0]["args"] == {"input": "edited"}
        
        # Verify additional_kwargs is synced
        raw_tool_calls = updated_ai_message.additional_kwargs["tool_calls"]
        assert len(raw_tool_calls) == 1
        assert raw_tool_calls[0]["function"]["arguments"] == '{"input": "edited"}'
        assert raw_tool_calls[0]["function"]["name"] == "test_tool"


def test_human_in_the_loop_middleware_syncs_additional_kwargs_with_name_change() -> None:
    """Test that HITL middleware syncs additional_kwargs during name edits."""
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={"test_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
    )

    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "test_tool", "args": {"input": "original"}, "id": "1"}],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "arguments": '{"input": "original"}',
                        "name": "test_tool",
                    },
                    "type": "function",
                }
            ]
        },
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])

    def mock_edit(_: Any) -> dict[str, Any]:
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": Action(
                        name="new_tool",
                        args={"input": "edited"},
                    ),
                }
            ]
        }

    with patch("langchain.agents.middleware.human_in_the_loop.interrupt", side_effect=mock_edit):
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "messages" in result
        updated_ai_message = result["messages"][0]
        
        # Verify tool_calls is updated
        assert updated_ai_message.tool_calls[0]["name"] == "new_tool"
        assert updated_ai_message.tool_calls[0]["args"] == {"input": "edited"}
        
        # Verify additional_kwargs is synced
        raw_tool_calls = updated_ai_message.additional_kwargs["tool_calls"]
        assert len(raw_tool_calls) == 1
        assert raw_tool_calls[0]["function"]["arguments"] == '{"input": "edited"}'
        assert raw_tool_calls[0]["function"]["name"] == "new_tool"

if __name__ == "__main__":
    test_human_in_the_loop_middleware_syncs_additional_kwargs()
    test_human_in_the_loop_middleware_syncs_additional_kwargs_with_name_change()
    print("All tests passed!")
