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
from langchain_glm.agent_toolkits.all_tools.tool import (
    AllToolExecutor,
    BaseToolOutput,
)

logger = logging.getLogger(__name__)


class CodeInterpreterToolOutput(BaseToolOutput):
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
class CodeInterpreterAllToolExecutor(AllToolExecutor):
    """platform adapter tool for code interpreter tool"""

    name: str

    @staticmethod
    def _python_ast_interpreter(
        code_input: str, platform_params: Dict[str, Any] = None
    ):
        """Use Shell to execute system shell commands"""

        try:
            from langchain_experimental.tools import PythonAstREPLTool

            tool = PythonAstREPLTool()
            out = tool.run(tool_input=code_input)

            return CodeInterpreterToolOutput(
                data=f"""Access：code_interpreter,{tool.name}, Message: {code_input}\r\n{out}""",
                platform_params=platform_params,
            )
        except ImportError:
            raise AttributeError(
                "This tool has been moved to langchain experiment. "
                "This tool has access to a python REPL. "
                "For best practices make sure to sandbox this tool. "
                "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
                "To keep using this code as is, install langchain experimental and "
                "update relevant imports replacing 'langchain' with 'langchain_experimental'"
            )

    def run(
        self,
        tool: str,
        tool_input: str,
        log: str,
        outputs: List[Union[str, dict]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> CodeInterpreterToolOutput:
        if outputs is None or str(outputs).strip() == "":
            if "auto" == self.platform_params.get("sandbox", "auto"):
                raise ValueError(
                    f"Tool {self.name} sandbox is auto , but log is None, is server error"
                )
            elif "none" == self.platform_params.get("sandbox", "auto"):
                logger.warning(
                    f"Tool {self.name} sandbox is local!!!, this not safe, please use jupyter sandbox it"
                )
                return self._python_ast_interpreter(
                    code_input=tool_input, platform_params=self.platform_params
                )

        return CodeInterpreterToolOutput(
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
    ) -> CodeInterpreterToolOutput:
        """Use the tool asynchronously."""
        if outputs is None or str(outputs).strip() == "" or len(outputs) == 0:
            if "auto" == self.platform_params.get("sandbox", "auto"):
                raise ValueError(
                    f"Tool {self.name} sandbox is auto , but log is None, is server error"
                )
            elif "none" == self.platform_params.get("sandbox", "auto"):
                logger.warning(
                    f"Tool {self.name} sandbox is local!!!, this not safe, please use jupyter sandbox it"
                )
                return self._python_ast_interpreter(
                    code_input=tool_input, platform_params=self.platform_params
                )

        return CodeInterpreterToolOutput(
            data=f"""Access：{tool}, Message: {tool_input},{log}""",
            platform_params=self.platform_params,
        )


class CodeInterpreterAdapterAllTool(AdapterAllTool[CodeInterpreterAllToolExecutor]):
    @classmethod
    def get_type(cls) -> str:
        return "CodeInterpreterAdapterAllTool"

    def _build_adapter_all_tool(
        self, platform_params: Dict[str, Any]
    ) -> CodeInterpreterAllToolExecutor:
        return CodeInterpreterAllToolExecutor(
            name="code_interpreter", platform_params=platform_params
        )
