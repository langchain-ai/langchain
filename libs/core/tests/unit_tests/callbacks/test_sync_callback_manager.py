from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool


def test_remove_handler() -> None:
    """Test removing handler does not raise an error on removal.

    An handler can be inheritable or not. This test checks that
    removing a handler does not raise an error if the handler
    is not inheritable.
    """
    handler1 = BaseCallbackHandler()
    handler2 = BaseCallbackHandler()
    manager = BaseCallbackManager([handler1], inheritable_handlers=[handler2])
    manager.remove_handler(handler1)
    manager.remove_handler(handler2)


def test_merge_preserves_handler_distinction() -> None:
    """Test that merging managers preserves the distinction between handlers.

    This test verifies the correct behavior of the BaseCallbackManager.merge()
    method. When two managers are merged, their handlers and
    inheritable_handlers should be combined independently.

    Currently, it is expected to xfail until the issue is resolved.
    """
    h1 = BaseCallbackHandler()
    h2 = BaseCallbackHandler()
    ih1 = BaseCallbackHandler()
    ih2 = BaseCallbackHandler()

    m1 = BaseCallbackManager(handlers=[h1], inheritable_handlers=[ih1])
    m2 = BaseCallbackManager(handlers=[h2], inheritable_handlers=[ih2])

    merged = m1.merge(m2)

    assert set(merged.handlers) == {h1, h2}
    assert set(merged.inheritable_handlers) == {ih1, ih2}


def test_tool_on_tool_end_preserves_artifact() -> None:
    """Test that on_tool_end receives ToolMessage with artifact.

    When a tool with response_format='content_and_artifact' is invoked
    without a tool_call_id, the on_tool_end callback should still receive
    a ToolMessage containing the artifact.
    """

    class ToolEndCapture(BaseCallbackHandler):
        """Captures on_tool_end output."""

        def __init__(self) -> None:
            super().__init__()
            self.outputs: list[Any] = []

        def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # noqa: ARG002
            self.outputs.append(output)

    @tool(response_format="content_and_artifact")
    def artifact_tool(text: str) -> tuple[str, dict[str, str]]:
        """A tool that returns content and artifact."""
        return text.upper(), {"original": text, "transformed": text.upper()}

    handler = ToolEndCapture()
    result = artifact_tool.invoke(
        {"text": "hello"},
        config={"callbacks": [handler]},
    )

    # The return value should be a ToolMessage with artifact
    assert isinstance(result, ToolMessage)
    assert result.content == "HELLO"
    assert result.artifact == {"original": "hello", "transformed": "HELLO"}
    assert result.tool_call_id == ""

    # The callback should have received the same ToolMessage
    assert len(handler.outputs) == 1
    callback_output = handler.outputs[0]
    assert isinstance(callback_output, ToolMessage)
    assert callback_output.content == "HELLO"
    assert callback_output.artifact == {"original": "hello", "transformed": "HELLO"}


def test_tool_on_tool_end_no_artifact_returns_string() -> None:
    """Test that tools without artifacts still return plain strings.

    When a tool does NOT use response_format='content_and_artifact',
    invoking without a tool_call_id should return the content as a string.
    """

    class ToolEndCapture(BaseCallbackHandler):
        """Captures on_tool_end output."""

        def __init__(self) -> None:
            super().__init__()
            self.outputs: list[Any] = []

        def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # noqa: ARG002
            self.outputs.append(output)

    @tool
    def simple_tool(text: str) -> str:
        """A simple tool."""
        return text.upper()

    handler = ToolEndCapture()
    result = simple_tool.invoke(
        {"text": "hello"},
        config={"callbacks": [handler]},
    )

    # Without content_and_artifact, the result should be a plain string
    assert isinstance(result, str)
    assert result == "HELLO"

    # The callback should receive the same string
    assert len(handler.outputs) == 1
    assert handler.outputs[0] == "HELLO"
