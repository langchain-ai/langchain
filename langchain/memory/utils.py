from typing import List

from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


def get_buffer_string(messages: List[BaseMessage]) -> str:
    """Get buffer string of messages."""
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "Human"
        elif isinstance(m, AIMessage):
            role = "AI"
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        string_messages.append(f"{role}: {m.content}")
    return "\n".join(string_messages)
