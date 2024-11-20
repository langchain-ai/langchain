from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool

from langchain_tests.unit_tests.tools import ToolsTests


class ToolsIntegrationTests(ToolsTests):
    def test_invoke_matches_output_schema(self, tool: BaseTool) -> None:
        """
        If invoked with a ToolCall, the tool should return a valid ToolMessage content.
        """
        tool_call = ToolCall(
            name=tool.name, args=self.tool_invoke_params_example, id=None
        )
        result = tool.invoke(tool_call)

        if tool.response_format == "content":
            content = result
        elif tool.response_format == "content_and_artifact":
            # should be (content, artifact)
            assert isinstance(result, tuple)
            assert len(result) == 2
            content, artifact = result

            assert artifact  # artifact can be anything, but shouldn't be none

        # check content is a valid ToolMessage content
        assert isinstance(content, (str, list))
        if isinstance(content, list):
            # content blocks must be str or dict
            assert all(isinstance(c, (str, dict)) for c in content)

    async def test_async_invoke_matches_output_schema(self, tool: BaseTool) -> None:
        """
        If ainvoked with a ToolCall, the tool should return a valid ToolMessage content.
        """
        tool_call = ToolCall(
            name=tool.name, args=self.tool_invoke_params_example, id=None
        )
        result = await tool.ainvoke(tool_call)

        if tool.response_format == "content":
            content = result
        elif tool.response_format == "content_and_artifact":
            # should be (content, artifact)
            assert isinstance(result, tuple)
            assert len(result) == 2
            content, artifact = result

            assert artifact  # artifact can be anything, but shouldn't be none

        # check content is a valid ToolMessage content
        assert isinstance(content, (str, list))
        if isinstance(content, list):
            # content blocks must be str or dict
            assert all(isinstance(c, (str, dict)) for c in content)

    def test_invoke_no_tool_call(self, tool: BaseTool) -> None:
        """
        If invoked without a ToolCall, the tool can return anything
        but it shouldn't throw an error
        """
        tool.invoke(self.tool_invoke_params_example)

    async def test_async_invoke_no_tool_call(self, tool: BaseTool) -> None:
        """
        If invoked without a ToolCall, the tool can return anything
        but it shouldn't throw an error
        """
        await tool.ainvoke(self.tool_invoke_params_example)
