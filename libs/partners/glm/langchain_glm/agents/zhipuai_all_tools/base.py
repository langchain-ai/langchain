import asyncio
import json
import logging
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain import hub
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic.v1 import BaseModel, Field, validator

from langchain_glm.agent_toolkits.all_tools.registry import (
    TOOL_STRUCT_TYPE_TO_TOOL_CLASS,
)
from langchain_glm.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_glm.agent_toolkits.all_tools.tool import (
    AdapterAllTool,
    BaseToolOutput,
)
from langchain_glm.agents.all_tools_agent import ZhipuAiAllToolsAgentExecutor
from langchain_glm.agents.all_tools_bind.base import create_zhipuai_tools_agent
from langchain_glm.agents.format_scratchpad.all_tools import (
    format_to_zhipuai_all_tool_messages,
)
from langchain_glm.agents.output_parsers import ZhipuAiALLToolsAgentOutputParser
from langchain_glm.agents.zhipuai_all_tools.schema import (
    AllToolsAction,
    AllToolsActionToolEnd,
    AllToolsActionToolStart,
    AllToolsFinish,
    AllToolsLLMStatus,
)
from langchain_glm.callbacks.agent_callback_handler import (
    AgentExecutorAsyncIteratorCallbackHandler,
    AgentStatus,
)
from langchain_glm.chat_models import ChatZhipuAI
from langchain_glm.utils import History

logger = logging.getLogger()


def _is_assistants_builtin_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> bool:
    """platform tools built-in"""
    assistants_builtin_tools = AdapterAllToolStructType.__members__.values()
    return (
        isinstance(tool, dict)
        and ("type" in tool)
        and (tool["type"] in assistants_builtin_tools)
    )


def _get_assistants_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an ZhipuAI tool."""
    if _is_assistants_builtin_tool(tool):
        return tool  # type: ignore
    else:
        # in case of a custom tool, convert it to an function of type
        return convert_to_openai_tool(tool)


def _agents_registry(
    llm: BaseLanguageModel,
    llm_with_all_tools: RunnableBindingBase = None,
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]] = [],
    callbacks: List[BaseCallbackHandler] = [],
    verbose: bool = False,
):
    if llm_with_all_tools:
        prompt = hub.pull("zhipuai-all-tools-chat/zhipuai-all-tools-agent")
        agent = create_zhipuai_tools_agent(
            prompt=prompt, llm_with_all_tools=llm_with_all_tools
        )
    else:
        prompt = hub.pull("zhipuai-all-tools-chat/zhipuai-all-tools-chat")
        agent = prompt | llm | ZhipuAiALLToolsAgentOutputParser()

    # AgentExecutor._aperform_agent_action = _aperform_agent_action
    # AgentExecutor._perform_agent_action = _perform_agent_action

    agent_executor = ZhipuAiAllToolsAgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        callbacks=callbacks,
        return_intermediate_steps=True,
    )

    return agent_executor


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        msg = f"Caught exception: {e}"
        logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)
    finally:
        # Signal the aiter to stop.
        event.set()


OutputType = Union[
    AllToolsAction,
    AllToolsActionToolStart,
    AllToolsActionToolEnd,
    AllToolsFinish,
    AllToolsLLMStatus,
]


class ZhipuAIAllToolsRunnable(RunnableSerializable[Dict, OutputType]):
    agent_executor: AgentExecutor
    """ZhipuAI AgentExecutor."""

    model_name: str = Field(default="tob-alltools-api-dev")
    """工具模型"""
    callback: AgentExecutorAsyncIteratorCallbackHandler
    """ZhipuAI AgentExecutor callback."""
    check_every_ms: float = 1_000.0
    """Frequency with which to check run progress in ms."""
    intermediate_steps: List[Tuple[AgentAction, BaseToolOutput]] = []
    """intermediate_steps to store the data to be processed."""
    history: List[Union[List, Tuple, Dict]] = []
    """user message history"""

    class Config:
        arbitrary_types_allowed = True

    @validator("intermediate_steps", pre=True, each_item=True, allow_reuse=True)
    def check_intermediate_steps(cls, v):
        return v

    @staticmethod
    def paser_all_tools(
        tool: Dict[str, Any], callbacks: List[BaseCallbackHandler] = []
    ) -> AdapterAllTool:
        platform_params = {}
        if tool["type"] in tool:
            platform_params = tool[tool["type"]]

        if tool["type"] in TOOL_STRUCT_TYPE_TO_TOOL_CLASS:
            all_tool = TOOL_STRUCT_TYPE_TO_TOOL_CLASS[tool["type"]](
                name=tool["type"], platform_params=platform_params, callbacks=callbacks
            )
            return all_tool
        else:
            raise ValueError(f"Unknown tool type: {tool['type']}")

    @classmethod
    def create_agent_executor(
        cls,
        model_name: str,
        *,
        intermediate_steps: List[Tuple[AgentAction, BaseToolOutput]] = [],
        history: List[Union[List, Tuple, Dict]] = [],
        tools: Sequence[
            Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]
        ] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> "ZhipuAIAllToolsRunnable":
        """Create an ZhipuAI Assistant and instantiate the Runnable."""

        callback = AgentExecutorAsyncIteratorCallbackHandler()
        callbacks = [callback]
        params = dict(
            streaming=True,
            verbose=True,
            callbacks=callbacks,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

        llm = ChatZhipuAI(**params)

        llm_with_all_tools = None

        temp_tools = []
        if tools:
            llm_with_all_tools = llm.bind(
                tools=[_get_assistants_tool(tool) for tool in tools]
            )

            temp_tools.extend(
                [
                    t.copy(update={"callbacks": callbacks})
                    for t in tools
                    if not _is_assistants_builtin_tool(t)
                ]
            )

            assistants_builtin_tools = []
            for t in tools:
                # TODO: platform tools built-in for all tools,
                #       load with langchain_glm/agents/all_tools_agent.py:108
                # AdapterAllTool implements it
                if _is_assistants_builtin_tool(t):
                    assistants_builtin_tools.append(cls.paser_all_tools(t, callbacks))
            temp_tools.extend(assistants_builtin_tools)

        agent_executor = _agents_registry(
            llm=llm,
            callbacks=callbacks,
            tools=temp_tools,
            llm_with_all_tools=llm_with_all_tools,
            verbose=True,
        )
        return cls(
            model_name=model_name,
            agent_executor=agent_executor,
            callback=callback,
            intermediate_steps=intermediate_steps,
            history=history,
            **kwargs,
        )

    def invoke(
        self, chat_input: str, config: Optional[RunnableConfig] = None
    ) -> AsyncIterable[OutputType]:
        async def chat_iterator() -> AsyncIterable[OutputType]:
            history_message = []
            if self.history:
                _history = [History.from_data(h) for h in self.history]
                chat_history = [h.to_msg_tuple() for h in _history]

                history_message = convert_to_messages(chat_history)

            task = asyncio.create_task(
                wrap_done(
                    self.agent_executor.ainvoke(
                        {
                            "input": chat_input,
                            "chat_history": history_message,
                            "agent_scratchpad": lambda x: format_to_zhipuai_all_tool_messages(
                                self.intermediate_steps
                            ),
                        }
                    ),
                    self.callback.done,
                )
            )

            async for chunk in self.callback.aiter():
                data = json.loads(chunk)
                class_status = None
                if data["status"] == AgentStatus.llm_start:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["text"],
                    )

                elif data["status"] == AgentStatus.llm_new_token:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["text"],
                    )
                elif data["status"] == AgentStatus.llm_end:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["text"],
                    )
                elif data["status"] == AgentStatus.agent_action:
                    class_status = AllToolsAction(
                        run_id=data["run_id"], status=data["status"], **data["action"]
                    )

                elif data["status"] == AgentStatus.tool_start:
                    class_status = AllToolsActionToolStart(
                        run_id=data["run_id"],
                        status=data["status"],
                        tool_input=data["tool_input"],
                        tool=data["tool"],
                    )

                elif data["status"] in [AgentStatus.tool_end]:
                    class_status = AllToolsActionToolEnd(
                        run_id=data["run_id"],
                        status=data["status"],
                        tool=data["tool"],
                        tool_output=data["tool_output"],
                    )
                elif data["status"] == AgentStatus.agent_finish:
                    class_status = AllToolsFinish(
                        run_id=data["run_id"],
                        status=data["status"],
                        **data["finish"],
                    )

                elif data["status"] == AgentStatus.agent_finish:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["outputs"]["output"],
                    )

                elif data["status"] == AgentStatus.error:
                    class_status = AllToolsLLMStatus(
                        run_id=data.get("run_id", "abc"),
                        status=data["status"],
                        text=json.dumps(data, ensure_ascii=False),
                    )
                elif data["status"] == AgentStatus.chain_start:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text="",
                    )
                elif data["status"] == AgentStatus.chain_end:
                    class_status = AllToolsLLMStatus(
                        run_id=data["run_id"],
                        status=data["status"],
                        text=data["outputs"]["output"],
                    )

                yield class_status

            await task

            if self.callback.out:
                self.history.append({"role": "user", "content": chat_input})
                self.history.append(
                    {"role": "assistant", "content": self.callback.outputs["output"]}
                )
                self.intermediate_steps.extend(self.callback.intermediate_steps)

        return chat_iterator()
