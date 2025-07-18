import uuid
from dataclasses import dataclass, field
from typing import TypedDict
from typing import Union, List, Optional, Literal, Any

from langchain_core.messages.ai import UsageMetadata


def _ensure_id(id_val: Optional[str]) -> str:
    """Ensure the ID is a valid string, generating a new UUID if not provided."""
    return id_val or str(uuid.uuid4())


@dataclass
class ContentBlock:
    """Replace with actual content blocks."""
    type: Literal["text", "image", "code", "other"]
    data: Union[str, dict]


class Provider(TypedDict):
    """Information about the provider that generated the message."""
    name: str
    """Name and version of the provider that created the content block."""
    model_name: str
    """Name of the model that generated the content block."""

@dataclass
class AIMessage:
    id: str
    type: Literal["ai"] = "ai"
    name: Optional[str] = None
    """An optional name for the message.

    This can be used to provide a human-readable name for the message.

    Usage of this field is optional, and whether it's used or not is up to the
    model implementation.
    """

    lc_version: str = "v1"
    """Encoding version for the message."""

    content: List[ContentBlock] = field(default_factory=list)
    tool_calls: Optional[List[dict]] = None
    invalid_tool_calls: Optional[List[dict]] = None
    usage: Optional[dict] = None

    def __post_init__(self) -> None:
        """Ensure fields."""
        self.id = _ensure_id(self.id)
        self._response = None

    @property
    def response(self) -> Optional[Any]:
        """Get the raw provider response when available.

        Response is expected to be serializable.
        """
        return self._response


@dataclass
class AIMessageChunk:
    id: str
    type: Literal["ai_chunk"] = "ai_chunk"
    name: Optional[str] = None
    """An optional name for the message.

    This can be used to provide a human-readable name for the message.

    Usage of this field is optional, and whether it's used or not is up to the
    model implementation.
    """
    content: List[ContentBlock] = field(default_factory=list)
    tool_call_chunks: Optional[List[dict]] = None

    usage_metadata: Optional[UsageMetadata] = None
    """If provided, usage metadata for a message, such as token counts.

    This is a standard representation of token usage that is consistent across models.
    """

    type: Literal["human"] = "human"
    """The type of the message. Must be a string that is unique to the message type.

    The purpose of this field is to allow for easy identification of the message type
    when deserializing messages.
    """

    def __post_init__(self) -> None:
        self.id = _ensure_id(self.id)


@dataclass
class HumanMessage:
    id: str
    content: List[ContentBlock]
    name: Optional[str] = None
    """An optional name for the message.

    This can be used to provide a human-readable name for the message.

    Usage of this field is optional, and whether it's used or not is up to the
    model implementation.
    """
    type: Literal["human"] = "human"
    """The type of the message. Must be a string that is unique to the message type.

    The purpose of this field is to allow for easy identification of the message type
    when deserializing messages.
    """

    def __init__(
        self, content: Union[str, List[ContentBlock]], id: Optional[str] = None
    ):
        self.id = _ensure_id(id)
        if isinstance(content, str):
            self.content = [ContentBlock(type="text", data=content)]
        else:
            self.content = content

    def text(self) -> str:
        return "".join(
            block.data
            for block in self.content
            if block.type == "text" and isinstance(block.data, str)
        )


@dataclass
class SystemMessage:
    id: str
    content: List[ContentBlock]
    type: Literal["system"] = "system"

    def __init__(
        self, content: Union[str, List[ContentBlock]], *, id: Optional[str] = None
    ):
        self.id = _ensure_id(id)
        if isinstance(content, str):
            self.content = [ContentBlock(type="text", data=content)]
        else:
            self.content = content

    def text(self) -> str:
        return "".join(
            block.data
            for block in self.content
            if block.type == "text" and isinstance(block.data, str)
        )


@dataclass
class ToolMessage:
    id: str
    tool_call_id: str
    content: Union[str, List[ContentBlock]]
    artifact: Optional[Any] = None  # App-side payload not for the model
    status: Literal["success", "error"] = "success"
    type: Literal["tool"] = "tool"

    def __post_init__(self):
        self.id = _ensure_id(self.id)


# Alias for a message type that can be any of the defined message types
Message = Union[
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
]
