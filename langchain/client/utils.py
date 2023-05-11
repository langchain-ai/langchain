"""Client Utils."""
from typing import Any, Dict

from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


def deserialize_message(message: Dict[str, Any]) -> BaseMessage:
    type_ = message.pop("_type")
    if type_ == "human":
        return HumanMessage(**message)
    elif type_ == "system":
        return SystemMessage(**message)
    elif type_ == "ai":
        return AIMessage(**message)
    elif type_ == "chat":
        return ChatMessage(**message)
    else:
        raise ValueError(f"Unknown message type: {type_}")
