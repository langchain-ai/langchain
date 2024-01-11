from typing import List, Optional, Union, Any, TypedDict

from langchain_core.runnables import RunnableSerializable, RunnableConfig
from langchain_core.tools import BaseTool, ToolInput


class ToolInvocation(TypedDict):
    tool_name: str
    tool_input: ToolInput


class ToolExecutor(RunnableSerializable[ToolInvocation, Any]):
    tools: List[BaseTool]

    def _get_tool(self, tool_name: str) -> BaseTool:
        tool_map = {tool.name: tool for tool in self.tools}
        if tool_name not in tool_map:
            raise ValueError
        return tool_map[tool_name]

    def invoke(self, input: ToolInvocation, config: Optional[RunnableConfig] = None) -> Any:
        tool = self._get_tool(input["tool_name"])
        return tool.invoke(input["tool_input"], config=config)

    async def ainvoke(self, input: ToolInvocation, config: Optional[RunnableConfig] = None) -> Any:
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
            batch_by_tool[input["tool_name"]] = batch_by_tool.get(input["tool_name"], []) + [input["tool_input"]]
        configs = get_config_list(config, len(inputs))
        with get_executor_for_config(configs[0]) as executor:
            return cast(
                List[Output],
                list(executor.map(invoke, runnables, actual_inputs, configs)),
            )




