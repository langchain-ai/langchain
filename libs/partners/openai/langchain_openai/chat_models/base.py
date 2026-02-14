from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections.abc import Mapping, Sequence
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

import openai
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


def _sanitize_text_content_block(content_block: dict[str, Any]) -> Union[str, dict[str, Any]]:
    """Sanitize a text content block by removing 'id' keys.
    
    OpenAI's API does not accept 'id' keys in text content blocks for ToolMessages.
    If the content block is a text block with an 'id' key, return just the text value.
    Otherwise, return the content block as-is.
    
    Args:
        content_block: A content block dictionary.
        
    Returns:
        Either a string (if it was a text block with id) or the original dict.
    """
    if isinstance(content_block, dict):
        # Check if this is a text content block with an 'id' key
        if content_block.get("type") == "text" and "id" in content_block:
            # Return just the text value, stripping the 'id'
            return {"type": "text", "text": content_block.get("text", "")}
    return content_block


def _convert_message_to_dict(
    message: BaseMessage,
) -> ChatCompletionMessageParam:
    """Convert a LangChain message to an OpenAI message dict.
    
    Args:
        message: The LangChain message to convert.
        
    Returns:
        An OpenAI-compatible message dictionary.
    """
    message_dict: dict[str, Any] = {"role": message.type}
    
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["args"]),
                    },
                }
                for tool_call in message.tool_calls
            ]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        # Sanitize content blocks to remove 'id' keys from text blocks
        content = message.content
        if isinstance(content, list):
            content = [_sanitize_text_content_block(block) for block in content]
        
        message_dict = {
            "role": "tool",
            "content": content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown message type: {message}")
    
    return cast("ChatCompletionMessageParam", message_dict)
