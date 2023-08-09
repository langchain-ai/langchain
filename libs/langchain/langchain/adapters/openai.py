from __future__ import annotations

from typing import List, Any, Mapping, Dict, Union

from langchain.schema.messages import HumanMessage, AIMessage, FunctionMessage, BaseMessageChunk, SystemMessage, BaseMessage, ChatMessage, HumanMessageChunk, ChatMessageChunk, SystemMessageChunk, FunctionMessageChunk, AIMessageChunk
import importlib


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def convert_openai_messages(messages: List[dict]) -> List[BaseMessage]:
    """Convert dictionaries representing OpenAI messages to LangChain format.

    Args:
        messages: List of dictionaries representing OpenAI messages

    Returns:
        List of LangChain BaseMessage objects.
    """
    return [convert_dict_to_message(m) for m in messages]


def _convert_message_chunk_to_delta(
        chunk: BaseMessageChunk
) -> Dict[str, Union[str, Any]]:
    _dict = {}

    # Determine role based on chunk type
    if isinstance(chunk, HumanMessageChunk):
        _dict["role"] = "user"
        _dict["content"] = chunk.content
    elif isinstance(chunk, AIMessageChunk):
        _dict["role"] = "assistant"
        _dict["content"] = chunk.content
        if hasattr(chunk, "function_call"):
            _dict["function_call"] = chunk.function_call
    elif isinstance(chunk, SystemMessageChunk):
        _dict["role"] = "system"
        _dict["content"] = chunk.content
    elif isinstance(chunk, FunctionMessageChunk):
        _dict["role"] = "function"
        _dict["content"] = chunk.content
        _dict["name"] = chunk.name
    elif isinstance(chunk, ChatMessageChunk):
        _dict["role"] = chunk.role
        _dict["content"] = chunk.content
    else:
        _dict["content"] = chunk.content

    return _dict

class ChatCompletion:

    @staticmethod
    def create(messages: List[dict], provider: str = "ChatOpenAI", stream:bool = False, **kwargs: Any,):
        models = importlib.import_module("langchain.chat_models")
        model_cls = getattr(models, provider)
        model_config = model_cls(**kwargs)
        messages = convert_openai_messages(messages)
        if not stream:
            result = model_config.invoke(messages)
            return {
                "choices": [
                    {"message": convert_message_to_dict(result)}
                ]
            }
        else:
            i = 0
            for c in model_config.stream(messages):

            return

    @staticmethod
    async def acreate(messages: List[dict], provider: str = "ChatOpenAI", stream:bool = False, **kwargs: Any,):
        models = importlib.import_module("langchain.chat_models")
        model_cls = getattr(models, provider)
        model_config = model_cls(**kwargs)
        messages = convert_openai_messages(messages)
        if not stream:
            result = await model_config.ainvoke(messages)
            return {
                "choices": [
                    {"message": convert_message_to_dict(result)}
                ]
            }
        else:
            return model_config.astream(messages)
