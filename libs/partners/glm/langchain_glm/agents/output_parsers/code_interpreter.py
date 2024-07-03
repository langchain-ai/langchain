import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Union

from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
)
from langchain_core.utils.json import (
    parse_partial_json,
)
from zhipuai.core import BaseModel

from langchain_glm.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_glm.agents.output_parsers._utils import (
    concatenate_segments,
    find_object_positions,
)
from langchain_glm.agents.output_parsers.base import (
    AllToolsMessageToolCall,
    AllToolsMessageToolCallChunk,
)
from langchain_glm.chat_models.all_tools_message import ALLToolsMessageChunk

logger = logging.getLogger(__name__)


class CodeInterpreterAgentAction(ToolAgentAction):
    outputs: List[Union[str, dict]] = None
    """Output of the tool call."""
    platform_params: dict = None


def _best_effort_parse_code_interpreter_tool_calls(
    tool_call_chunks: List[dict],
) -> List[Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]]:
    code_interpreter_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    # Best-effort parsing allready parsed tool calls
    for code_interpreter in tool_call_chunks:
        if AdapterAllToolStructType.CODE_INTERPRETER == code_interpreter["name"]:
            if isinstance(code_interpreter["args"], str):
                args_ = parse_partial_json(code_interpreter["args"])
            else:
                args_ = code_interpreter["args"]
            if not isinstance(args_, dict):
                raise ValueError("Malformed args.")

            if "outputs" in args_:
                code_interpreter_chunk.append(
                    AllToolsMessageToolCall(
                        name=code_interpreter["name"],
                        args=args_,
                        id=code_interpreter["id"],
                    )
                )
            else:
                code_interpreter_chunk.append(
                    AllToolsMessageToolCallChunk(
                        name=code_interpreter["name"],
                        args=args_,
                        id=code_interpreter["id"],
                        index=code_interpreter.get("index"),
                    )
                )

    return code_interpreter_chunk


def _paser_code_interpreter_chunk_input(
    message: BaseMessage,
    code_interpreter_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ],
) -> deque[CodeInterpreterAgentAction]:
    try:
        input_log_chunk = []

        outputs: List[List[dict]] = []
        obj = object()
        for interpreter_chunk in code_interpreter_chunk:
            interpreter_chunk_args = interpreter_chunk.args

            if "input" in interpreter_chunk_args:
                input_log_chunk.append(interpreter_chunk_args["input"])
            if "outputs" in interpreter_chunk_args:
                input_log_chunk.append(obj)
                outputs.append(interpreter_chunk_args["outputs"])

        if input_log_chunk[-1] is not obj:
            input_log_chunk.append(obj)
        # segments the list based on these positions, and then concatenates each segment into a string
        # Find positions of object() instances
        positions = find_object_positions(input_log_chunk, obj)

        # Concatenate segments
        result_actions = concatenate_segments(input_log_chunk, positions)

        tool_call_id = (
            code_interpreter_chunk[0].id if code_interpreter_chunk[0].id else "abc"
        )
        code_interpreter_action_result_stack: deque = deque()
        for i, action in enumerate(result_actions):
            if len(result_actions) > len(outputs):
                outputs.insert(i, [])

            out_logs = [logs["logs"] for logs in outputs[i] if "logs" in logs]
            out_str = "\n".join(out_logs)
            log = f"{action}\r\n{out_str}"
            code_interpreter_action = CodeInterpreterAgentAction(
                tool=AdapterAllToolStructType.CODE_INTERPRETER,
                tool_input=action,
                outputs=outputs[i],
                log=log,
                message_log=[message],
                tool_call_id=tool_call_id,
            )

            code_interpreter_action_result_stack.append(code_interpreter_action)
        return code_interpreter_action_result_stack

    except Exception as e:
        logger.error(f"Error parsing code_interpreter_chunk: {e}", exc_info=True)
        raise OutputParserException(
            f"Could not parse tool input: code_interpreter because "
            f"the `arguments` is not valid JSON."
        )
