from typing import Any, List, Optional, TypedDict, Union

from langchain_core.runnables import (
    RunnableConfig,
    RunnableSerializable,
    get_config_list,
)
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool, ToolInput


class ToolInvocation(TypedDict):
    tool_name: str
    tool_input: ToolInput


def _batch(tool, tool_inputs, config, return_exceptions):
    return tool.batch(tool_inputs, config=config, return_exceptions=return_exceptions)


class ToolExecutor(RunnableSerializable[ToolInvocation, Any]):
    tools: List[BaseTool]

    def _get_tool(self, tool_name: str) -> BaseTool:
        tool_map = {tool.name: tool for tool in self.tools}
        if tool_name not in tool_map:
            raise ValueError
        return tool_map[tool_name]

    def invoke(
        self, input: ToolInvocation, config: Optional[RunnableConfig] = None
    ) -> Any:
        tool = self._get_tool(input["tool_name"])
        return tool.invoke(input["tool_input"], config=config)

    async def ainvoke(
        self, input: ToolInvocation, config: Optional[RunnableConfig] = None
    ) -> Any:
        tool = self._get_tool(input["tool_name"])
        return await tool.ainvoke(input["tool_input"], config=config)

    def batch(
        self,
        inputs: List[ToolInvocation],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Any]:
        batch_by_tool = {}
        for input in inputs:
            batch_by_tool[input["tool_name"]] = batch_by_tool.get(
                input["tool_name"], []
            ) + [input["tool_input"]]
        tools = list(batch_by_tool.keys())
        tools_inputs = list(batch_by_tool.values())
        configs = get_config_list(config, len(tools))
        return_exceptions_list = [return_exceptions] * len(tools)
        with get_executor_for_config(configs[0]) as executor:
            return (
                list(
                    executor.map(
                        _batch, tools, tools_inputs, configs, return_exceptions_list
                    )
                ),
            )
