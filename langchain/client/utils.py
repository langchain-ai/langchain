"""Client Utils."""
import re
from typing import Dict, List, Optional, Sequence, Type, Union

from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

_DEFAULT_MESSAGES_T = Union[Type[HumanMessage], Type[SystemMessage], Type[AIMessage]]
_RESOLUTION_MAP: Dict[str, _DEFAULT_MESSAGES_T] = {
    "Human": HumanMessage,
    "AI": AIMessage,
    "System": SystemMessage,
}


def parse_chat_messages(
    input_text: str, roles: Optional[Sequence[str]] = None
) -> List[BaseMessage]:
    """Parse chat messages from a string. This is not robust."""
    roles = roles or ["Human", "AI", "System"]
    roles_pattern = "|".join(roles)
    pattern = (
        rf"(?P<entity>{roles_pattern}): (?P<message>"
        rf"(?:.*\n?)*?)(?=(?:{roles_pattern}): |\Z)"
    )
    matches = re.finditer(pattern, input_text, re.MULTILINE)

    results: List[BaseMessage] = []
    for match in matches:
        entity = match.group("entity")
        message = match.group("message").rstrip("\n")
        if entity in _RESOLUTION_MAP:
            results.append(_RESOLUTION_MAP[entity](content=message))
        else:
            results.append(ChatMessage(role=entity, content=message))

    return results
