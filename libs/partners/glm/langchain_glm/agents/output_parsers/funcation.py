import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Union

from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    BaseMessage,
    ToolCall,
)

logger = logging.getLogger(__name__)


def _paser_function_chunk_input(
    message: BaseMessage,
    function_chunk: List[ToolCall],
) -> deque[ToolAgentAction]:
    try:
        function_action_result_stack: deque = deque()
        for tool_call in function_chunk:
            # HACK HACK HACK:
            # The code that encodes tool input into Open AI uses a special variable
            # name called `__arg1` to handle old style tools that do not expose a
            # schema and expect a single string argument as an input.
            # We unpack the argument here if it exists.
            # Open AI does not support passing in a JSON array as an argument.
            function_name = tool_call["name"]
            _tool_input = tool_call["args"]
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"

            function_action_result_stack.append(
                ToolAgentAction(
                    tool=function_name,
                    tool_input=tool_input,
                    log=log,
                    message_log=[message],
                    tool_call_id=tool_call["id"] if tool_call["id"] else "abc",
                )
            )
        return function_action_result_stack
    except Exception as e:
        logger.error(f"Error parsing function_chunk: {e}", exc_info=True)
        raise OutputParserException(
            f"Error parsing function_chunk: {e} " f"the `arguments` is not valid JSON."
        )
