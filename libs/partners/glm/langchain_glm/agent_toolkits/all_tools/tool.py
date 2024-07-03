"""platform adapter tool """

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from dataclasses_json import DataClassJsonMixin
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain_core.tools import BaseTool

from langchain_glm.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_glm.agents.output_parsers.code_interpreter import (
    CodeInterpreterAgentAction,
)
from langchain_glm.agents.output_parsers.drawing_tool import DrawingToolAgentAction
from langchain_glm.agents.output_parsers.web_browser import WebBrowserAgentAction

logger = logging.getLogger(__name__)


class BaseToolOutput:
    """
    LLM 要求 Tool 的输出为 str，但 Tool 用在别处时希望它正常返回结构化数据。
    只需要将 Tool 返回值用该类封装，能同时满足两者的需要。
    基类简单的将返回值字符串化，或指定 format="json" 将其转为 json。
    用户也可以继承该类定义自己的转换方法。
    """

    def __init__(
        self,
        data: Any,
        format: str = "",
        data_alias: str = "",
        **extras: Any,
    ) -> None:
        self.data = data
        self.format = format
        self.extras = extras
        if data_alias:
            setattr(self, data_alias, property(lambda obj: obj.data))

    def __str__(self) -> str:
        if self.format == "json":
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        else:
            return str(self.data)


@dataclass
class AllToolExecutor(DataClassJsonMixin):
    platform_params: Dict[str, Any]

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> BaseToolOutput:
        pass

    @abstractmethod
    async def arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseToolOutput:
        pass


E = TypeVar("E", bound=AllToolExecutor)


class AdapterAllTool(BaseTool, Generic[E]):
    """platform adapter tool for all tools."""

    name: str
    description: str

    platform_params: Dict[str, Any]
    """tools params """
    adapter_all_tool: E

    def __init__(self, name: str, platform_params: Dict[str, Any], **data: Any):
        super().__init__(
            name=name,
            description=f"platform adapter tool for {name}",
            platform_params=platform_params,
            adapter_all_tool=self._build_adapter_all_tool(platform_params),
            **data,
        )

    @abstractmethod
    def _build_adapter_all_tool(self, platform_params: Dict[str, Any]) -> E:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        raise NotImplementedError

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        # For backwards compatibility, if run_input is a string,
        # pass as a positional argument.
        if tool_input is None:
            return (), {}
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            # for tool defined with `*args` parameters
            # the args_schema has a field named `args`
            # it should be expanded to actual *args
            # e.g.: test_tools
            #       .test_named_tool_decorator_return_direct
            #       .search_api
            if "args" in tool_input:
                args = tool_input["args"]
                if args is None:
                    tool_input.pop("args")
                    return (), tool_input
                elif isinstance(args, tuple):
                    tool_input.pop("args")
                    return args, tool_input
            return (), tool_input

    def _run(
        self,
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        **tool_run_kwargs: Any,
    ) -> Any:
        if (
            AdapterAllToolStructType.CODE_INTERPRETER == agent_action.tool
            and isinstance(agent_action, CodeInterpreterAgentAction)
        ):
            return self.adapter_all_tool.run(
                **{
                    "tool": agent_action.tool,
                    "tool_input": agent_action.tool_input,
                    "log": agent_action.log,
                    "outputs": agent_action.outputs,
                },
                **tool_run_kwargs,
            )
        elif AdapterAllToolStructType.DRAWING_TOOL == agent_action.tool and isinstance(
            agent_action, DrawingToolAgentAction
        ):
            return self.adapter_all_tool.run(
                **{
                    "tool": agent_action.tool,
                    "tool_input": agent_action.tool_input,
                    "log": agent_action.log,
                    "outputs": agent_action.outputs,
                },
                **tool_run_kwargs,
            )
        elif AdapterAllToolStructType.WEB_BROWSER == agent_action.tool and isinstance(
            agent_action, WebBrowserAgentAction
        ):
            return self.adapter_all_tool.run(
                **{
                    "tool": agent_action.tool,
                    "tool_input": agent_action.tool_input,
                    "log": agent_action.log,
                    "outputs": agent_action.outputs,
                },
                **tool_run_kwargs,
            )
        else:
            raise KeyError()

    async def _arun(
        self,
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        **tool_run_kwargs: Any,
    ) -> Any:
        if (
            AdapterAllToolStructType.CODE_INTERPRETER == agent_action.tool
            and isinstance(agent_action, CodeInterpreterAgentAction)
        ):
            return await self.adapter_all_tool.arun(
                **{
                    "tool": agent_action.tool,
                    "tool_input": agent_action.tool_input,
                    "log": agent_action.log,
                    "outputs": agent_action.outputs,
                },
                **tool_run_kwargs,
            )

        elif AdapterAllToolStructType.DRAWING_TOOL == agent_action.tool and isinstance(
            agent_action, DrawingToolAgentAction
        ):
            return await self.adapter_all_tool.arun(
                **{
                    "tool": agent_action.tool,
                    "tool_input": agent_action.tool_input,
                    "log": agent_action.log,
                    "outputs": agent_action.outputs,
                },
                **tool_run_kwargs,
            )
        elif AdapterAllToolStructType.WEB_BROWSER == agent_action.tool and isinstance(
            agent_action, WebBrowserAgentAction
        ):
            return await self.adapter_all_tool.arun(
                **{
                    "tool": agent_action.tool,
                    "tool_input": agent_action.tool_input,
                    "log": agent_action.log,
                    "outputs": agent_action.outputs,
                },
                **tool_run_kwargs,
            )
        else:
            raise KeyError()
