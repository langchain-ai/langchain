import json
import logging
from collections import deque
from json import JSONDecodeError
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


class DrawingToolAgentAction(ToolAgentAction):
    outputs: List[Union[str, dict]] = None
    """Output of the tool call."""
    platform_params: dict = None


def _best_effort_parse_drawing_tool_tool_calls(
    tool_call_chunks: List[dict],
) -> List[Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]]:
    drawing_tool_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ] = []
    # Best-effort parsing allready parsed tool calls
    for drawing_tool in tool_call_chunks:
        if AdapterAllToolStructType.DRAWING_TOOL == drawing_tool["name"]:
            if isinstance(drawing_tool["args"], str):
                args_ = parse_partial_json(drawing_tool["args"])
            else:
                args_ = drawing_tool["args"]
            if not isinstance(args_, dict):
                raise ValueError("Malformed args.")

            if "outputs" in args_:
                drawing_tool_chunk.append(
                    AllToolsMessageToolCall(
                        name=drawing_tool["name"],
                        args=args_,
                        id=drawing_tool["id"],
                    )
                )
            else:
                drawing_tool_chunk.append(
                    AllToolsMessageToolCallChunk(
                        name=drawing_tool["name"],
                        args=args_,
                        id=drawing_tool["id"],
                        index=drawing_tool.get("index"),
                    )
                )

    return drawing_tool_chunk


def _paser_drawing_tool_chunk_input(
    message: BaseMessage,
    drawing_tool_chunk: List[
        Union[AllToolsMessageToolCall, AllToolsMessageToolCallChunk]
    ],
) -> deque[DrawingToolAgentAction]:
    try:
        input_log_chunk = []

        outputs: List[List[dict]] = []
        obj = object()
        for interpreter_chunk in drawing_tool_chunk:
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

        tool_call_id = drawing_tool_chunk[0].id if drawing_tool_chunk[0].id else "abc"
        drawing_tool_action_result_stack: deque = deque()
        for i, action in enumerate(result_actions):
            if len(result_actions) > len(outputs):
                outputs.insert(i, [])

            out_logs = [
                f'<img src="{logs.get("image")}" width="300">'
                for logs in outputs[i]
                if "image" in logs
            ]

            out_str = "\n".join(out_logs)
            log = f"{action}\r\n{out_str}"

            drawing_tool_action = DrawingToolAgentAction(
                tool=AdapterAllToolStructType.DRAWING_TOOL,
                tool_input=action,
                outputs=outputs[i],
                log=log,
                message_log=[message],
                tool_call_id=tool_call_id,
            )
            drawing_tool_action_result_stack.append(drawing_tool_action)
        return drawing_tool_action_result_stack

    except Exception as e:
        logger.error(f"Error parsing drawing_tool_chunk: {e}", exc_info=True)
        raise OutputParserException(
            f"Could not parse tool input: drawing_tool because "
            f"the `arguments` is not valid JSON."
        )
