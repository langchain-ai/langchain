"""Client Utils."""
import re
from typing import List, Optional, Sequence

from langchain.schema import ChatMessage, HumanMessage, SystemMessage, AIMessage


_RESOLUTION_MAP = {
    "Human": HumanMessage,
    "AI": AIMessage,
    "System": SystemMessage,
}


def parse_chat_messages(
    input_text: str, roles: Optional[Sequence[str]] = None
) -> List[ChatMessage]:
    """Parse chat messages from a string. This is not robust."""
    roles = roles or ["Human", "AI", "System"]
    roles_pattern = "|".join(roles)
    pattern = rf"(?P<entity>{roles_pattern}): (?P<message>(?:.*\n?)*?)(?=(?:{roles_pattern}): |\Z)"
    matches = re.finditer(pattern, input_text, re.MULTILINE)

    results = []
    for match in matches:
        entity = match.group("entity")
        message = match.group("message").rstrip("\n")
        if entity in _RESOLUTION_MAP:
            results.append(_RESOLUTION_MAP[entity](content=message))
        else:
            results.append(ChatMessage(role=entity, content=message))

    return results
