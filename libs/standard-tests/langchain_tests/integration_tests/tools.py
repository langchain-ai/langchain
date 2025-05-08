from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool

from langchain_tests.unit_tests.tools import ToolsTests


class ToolsIntegrationTests(ToolsTests):
    """
    Base class for tools integration tests.
    """

    def test_invoke_matches_output_schema(self, tool: BaseTool) -> None:
        """
        If invoked with a ToolCall, the tool should return a valid ToolMessage content.

        If you have followed the `custom tool guide <https://python.langchain.com/docs/how_to/custom_tools/>`_,
        this test should always pass because ToolCall inputs are handled by the
        :class:`langchain_core.tools.BaseTool` class.

        If you have not followed this guide, you should ensure that your tool's
        `invoke` method returns a valid ToolMessage content when it receives
        a dict representing a ToolCall as input (as opposed to distinct args).
        """
        tool_call = ToolCall(
            name=tool.name,
            args=self.tool_invoke_params_example,
            id="123",
            type="tool_call",
        )
        result = tool.invoke(tool_call)

        tool_message = result
        if tool.response_format == "content_and_artifact":
            # artifact can be anything, except none
            assert tool_message.artifact is not None

        # check content is a valid ToolMessage content
        assert isinstance(tool_message.content, (str, list))
        if isinstance(tool_message.content, list):
            # content blocks must be str or dict
            assert all(isinstance(c, (str, dict)) for c in tool_message.content)

    async def test_async_invoke_matches_output_schema(self, tool: BaseTool) -> None:
        """
        If ainvoked with a ToolCall, the tool should return a valid ToolMessage content.

        For debugging tips, see :meth:`test_invoke_matches_output_schema`.
        """
        tool_call = ToolCall(
            name=tool.name,
            args=self.tool_invoke_params_example,
            id="123",
            type="tool_call",
        )
        result = await tool.ainvoke(tool_call)

        tool_message = result
        if tool.response_format == "content_and_artifact":
            # artifact can be anything, except none
            assert tool_message.artifact is not None

        # check content is a valid ToolMessage content
        assert isinstance(tool_message.content, (str, list))
        if isinstance(tool_message.content, list):
            # content blocks must be str or dict
            assert all(isinstance(c, (str, dict)) for c in tool_message.content)

    def test_invoke_no_tool_call(self, tool: BaseTool) -> None:
        """
        If invoked without a ToolCall, the tool can return anything
        but it shouldn't throw an error

        If this test fails, your tool may not be handling the input you defined
        in `tool_invoke_params_example` correctly, and it's throwing an error.

        This test doesn't have any checks. It's just to ensure that the tool
        doesn't throw an error when invoked with a dictionary of kwargs.
        """
        tool.invoke(self.tool_invoke_params_example)

    async def test_async_invoke_no_tool_call(self, tool: BaseTool) -> None:
        """
        If ainvoked without a ToolCall, the tool can return anything
        but it shouldn't throw an error

        For debugging tips, see :meth:`test_invoke_no_tool_call`.
        """
        await tool.ainvoke(self.tool_invoke_params_example)
