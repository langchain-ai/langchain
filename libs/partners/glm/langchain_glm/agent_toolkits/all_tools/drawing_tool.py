import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain_glm.agent_toolkits import AdapterAllTool
from langchain_glm.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_glm.agent_toolkits.all_tools.tool import (
    AllToolExecutor,
    BaseToolOutput,
)

logger = logging.getLogger(__name__)


class DrawingToolOutput(BaseToolOutput):
    platform_params: Dict[str, Any]

    def __init__(
        self,
        data: Any,
        platform_params: Dict[str, Any],
        **extras: Any,
    ) -> None:
        super().__init__(data, "", "", **extras)
        self.platform_params = platform_params


@dataclass
class DrawingAllToolExecutor(AllToolExecutor):
    """platform adapter tool for code interpreter tool"""

    name: str

    def run(
        self,
        tool: str,
        tool_input: str,
        log: str,
        outputs: List[Union[str, dict]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> DrawingToolOutput:
        if outputs is None or str(outputs).strip() == "":
            raise ValueError(f"Tool {self.name}  is server error")

        return DrawingToolOutput(
            data=f"""Access：{tool}, Message: {tool_input},{log}""",
            platform_params=self.platform_params,
        )

    async def arun(
        self,
        tool: str,
        tool_input: str,
        log: str,
        outputs: List[Union[str, dict]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> DrawingToolOutput:
        """Use the tool asynchronously."""
        if outputs is None or str(outputs).strip() == "" or len(outputs) == 0:
            raise ValueError(f"Tool {self.name}  is server error")

        return DrawingToolOutput(
            data=f"""Access：{tool}, Message: {tool_input},{log}""",
            platform_params=self.platform_params,
        )


class DrawingAdapterAllTool(AdapterAllTool[DrawingAllToolExecutor]):
    @classmethod
    def get_type(cls) -> str:
        return "DrawingAdapterAllTool"

    def _build_adapter_all_tool(
        self, platform_params: Dict[str, Any]
    ) -> DrawingAllToolExecutor:
        return DrawingAllToolExecutor(
            name=AdapterAllToolStructType.DRAWING_TOOL, platform_params=platform_params
        )
