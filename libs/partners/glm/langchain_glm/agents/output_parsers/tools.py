import json
import logging
from collections import deque
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
    ToolCallChunk,
)
from langchain_core.utils.json import (
    parse_partial_json,
)
from zhipuai.core import BaseModel

from langchain_glm.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_glm.agents.output_parsers.base import (
    AllToolsMessageToolCall,
    AllToolsMessageToolCallChunk,
)
from langchain_glm.agents.output_parsers.code_interpreter import (
    _best_effort_parse_code_interpreter_tool_calls,
    _paser_code_interpreter_chunk_input,
)
from langchain_glm.agents.output_parsers.drawing_tool import (
    _best_effort_parse_drawing_tool_tool_calls,
    _paser_drawing_tool_chunk_input,
)
from langchain_glm.agents.output_parsers.funcation import (
    _paser_function_chunk_input,
)
from langchain_glm.agents.output_parsers.web_browser import (
    _best_effort_parse_web_browser_tool_calls,
    _paser_web_browser_chunk_input,
)
from langchain_glm.chat_models.all_tools_message import ALLToolsMessageChunk

logger = logging.getLogger(__name__)


def parse_ai_message_to_tool_action(
    message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    # TODO: parse platform tools built-in @langchain_glm.agents.zhipuai_all_tools.base._get_assistants_tool
    #   type in the future "function" or "code_interpreter"
    #   for @ToolAgentAction from langchain.agents.output_parsers.tools
    #   import with langchain.agents.format_scratchpad.tools.format_to_tool_messages
    actions: List = []
    if message.tool_calls:
        tool_calls = message.tool_calls
    else:
        if not message.additional_kwargs.get("tool_calls"):
            return AgentFinish(
                return_values={"output": message.content}, log=str(message.content)
            )
        # Best-effort parsing allready parsed tool calls
        tool_calls = []
        for tool_call in message.additional_kwargs["tool_calls"]:
            if "function" == tool_call["type"]:
                function = tool_call["function"]
                function_name = function["name"]
                try:
                    args = json.loads(function["arguments"] or "{}")
                    tool_calls.append(
                        ToolCall(
                            name=function_name,
                            args=args,
                            id=tool_call["id"] if tool_call["id"] else "abc",
                        )
                    )
                except JSONDecodeError:
                    raise OutputParserException(
                        f"Could not parse tool input: {function} because "
                        f"the `arguments` is not valid JSON."
                    )
            elif tool_call["type"] in AdapterAllToolStructType.__members__.values():
                adapter_tool = tool_call[tool_call["type"]]

                tool_calls.append(
                    ToolCall(
                        name=tool_call["type"],
                        args=adapter_tool if adapter_tool else {},
                        id=tool_call["id"] if tool_call["id"] else "abc",
                    )
                )

    code_interpreter_action_result_stack: deque = deque()
    web_browser_action_result_stack: deque = deque()
    drawing_tool_result_stack: deque = deque()
    function_tool_result_stack: deque = deque()
    code_interpreter_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    if message.tool_calls:
        if isinstance(message, ALLToolsMessageChunk):
            code_interpreter_chunk = _best_effort_parse_code_interpreter_tool_calls(
                message.tool_call_chunks
            )
    else:
        code_interpreter_chunk = _best_effort_parse_code_interpreter_tool_calls(
            tool_calls
        )

    if code_interpreter_chunk and len(code_interpreter_chunk) > 1:
        code_interpreter_action_result_stack = _paser_code_interpreter_chunk_input(
            message, code_interpreter_chunk
        )

    drawing_tool_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    if message.tool_calls:
        if isinstance(message, ALLToolsMessageChunk):
            drawing_tool_chunk = _best_effort_parse_drawing_tool_tool_calls(
                message.tool_call_chunks
            )
    else:
        drawing_tool_chunk = _best_effort_parse_drawing_tool_tool_calls(tool_calls)

    if drawing_tool_chunk and len(drawing_tool_chunk) > 1:
        drawing_tool_result_stack = _paser_drawing_tool_chunk_input(
            message, drawing_tool_chunk
        )

    web_browser_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    if message.tool_calls:
        if isinstance(message, ALLToolsMessageChunk):
            web_browser_chunk = _best_effort_parse_web_browser_tool_calls(
                message.tool_call_chunks
            )
    else:
        web_browser_chunk = _best_effort_parse_web_browser_tool_calls(tool_calls)

    if web_browser_chunk and len(web_browser_chunk) > 1:
        web_browser_action_result_stack = _paser_web_browser_chunk_input(
            message, web_browser_chunk
        )

    # TODO: parse platform tools built-in @langchain_glm
    # delete AdapterAllToolStructType from tool_calls
    function_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if tool_call["name"] not in AdapterAllToolStructType.__members__.values()
    ]

    function_tool_result_stack = _paser_function_chunk_input(
        message, function_tool_calls
    )

    if isinstance(message, ALLToolsMessageChunk):
        call_chunks = _paser_object_positions(message.tool_call_chunks)

        for too_call_name in call_chunks:
            if too_call_name == AdapterAllToolStructType.CODE_INTERPRETER:
                actions.append(code_interpreter_action_result_stack.popleft())
            elif too_call_name == AdapterAllToolStructType.WEB_BROWSER:
                actions.append(web_browser_action_result_stack.popleft())
            elif too_call_name == AdapterAllToolStructType.DRAWING_TOOL:
                actions.append(drawing_tool_result_stack.popleft())
            else:
                actions.append(function_tool_result_stack.popleft())
    else:
        for too_call in tool_calls:
            if "function" == too_call["name"]:
                actions.append(function_tool_result_stack.popleft())
            elif too_call["name"] == AdapterAllToolStructType.CODE_INTERPRETER:
                actions.append(code_interpreter_action_result_stack.popleft())
            elif too_call["name"] == AdapterAllToolStructType.WEB_BROWSER:
                actions.append(web_browser_action_result_stack.popleft())
            elif too_call["name"] == AdapterAllToolStructType.DRAWING_TOOL:
                actions.append(drawing_tool_result_stack.popleft())

    return actions


def _paser_object_positions(tool_call_chunks: List[ToolCallChunk]):
    call_chunks = []
    last_name = None
    if not tool_call_chunks:
        return call_chunks
    for call_chunk in tool_call_chunks:
        if call_chunk["name"] in AdapterAllToolStructType.__members__.values():
            if isinstance(call_chunk["args"], str):
                args_ = parse_partial_json(call_chunk["args"])
            else:
                args_ = call_chunk["args"]
            if not isinstance(args_, dict):
                raise ValueError("Malformed args.")

            if "outputs" in args_:
                call_chunks.append(call_chunk["name"])
                last_name = call_chunk["name"]

        else:
            if call_chunk["name"] != last_name:
                call_chunks.append(call_chunk["name"])
                last_name = call_chunk["name"]

    if len(call_chunks) == 0:
        call_chunks.append(tool_call_chunks[-1]["name"])
    elif tool_call_chunks[-1]["name"] != call_chunks[-1]:
        call_chunks.append(tool_call_chunks[-1]["name"])
    return call_chunks
