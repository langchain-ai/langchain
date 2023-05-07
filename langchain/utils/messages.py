from langchain.schema import BaseMessage


def serialize_msgs(msgs: list[BaseMessage], include_type=False) -> str:
    return "\n\n".join(
        (f"{msg.type}: {msg.content}" if include_type else msg.content) for msg in msgs
    )
