from typing import Any, Sequence

from langchain_core.runnables.base import RunnableBindingBase, RunnableLambda
from langchain_core.tools import BaseTool, ToolInvocation


class InvalidToolNameException(Exception):
    def __init__(
        self,
        requested: str,
        available: Sequence[str],
    ) -> None:
        self.requested = requested
        self.available = available

    def __str__(self) -> str:
        available_str = ", ".join(t for t in self.available)
        return (
            f"{self.__class__.__name__}: Requested tool {self.requested} does not "
            f"exist. Available tools are {available_str}."
        )


class ToolExecutor(RunnableBindingBase):
    def __init__(self, tools: Sequence[BaseTool], **kwargs: Any) -> None:
        self._tools = tools
        self._tool_map = {t.name: t for t in self._tools}
        bound = RunnableLambda(self._execute, afunc=self._aexecute)
        super().__init__(bound=bound, **kwargs)

    def _execute(self, tool_invocation: ToolInvocation) -> Any:
        if tool_invocation.tool not in self._tool_map:
            exception = InvalidToolNameException(
                requested=tool_invocation.tool,
                available=list(self._tool_map.keys()),
            )
            return tool_invocation, exception
        else:
            tool = self._tool_map[tool_invocation.tool]
            output = tool.invoke(tool_invocation.tool_input)
            return tool_invocation, output

    async def _aexecute(self, tool_invocation: ToolInvocation) -> Any:
        if tool_invocation.tool not in self._tool_map:
            exception = InvalidToolNameException(
                requested=tool_invocation.tool,
                available=list(self._tool_map.keys()),
            )
            return tool_invocation, exception
        else:
            tool = self._tool_map[tool_invocation.tool]
            output = await tool.ainvoke(tool_invocation.tool_input)
            return tool_invocation, output
