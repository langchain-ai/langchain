from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

from typing_extensions import Literal

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Extra, Field

if TYPE_CHECKING:
    from langchain.prompts.chat import ChatPromptTemplate


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Convert sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
        ai_prefix: THe prefix to prepend to contents of AIMessages.

    Returns:
        A single string concatenation of all input messages.

    Example:
        .. code-block:: python

            from langchain.schema import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?\nAI: Good, how are you?"
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}: {m.content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)


class BaseMessage(Serializable):
    """The base abstract Message class.

    Messages are the inputs and outputs of ChatModels.
    """

    content: Union[str, List[Union[str, Dict]]]
    """The string contents of the message."""

    additional_kwargs: dict = Field(default_factory=dict)
    """Any additional information."""

    type: str

    class Config:
        extra = Extra.allow

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    def __add__(self, other: Any) -> ChatPromptTemplate:
        from langchain.prompts.chat import ChatPromptTemplate

        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other


def merge_content(
    first_content: Union[str, List[Union[str, Dict]]],
    second_content: Union[str, List[Union[str, Dict]]],
) -> Union[str, List[Union[str, Dict]]]:
    # If first chunk is a string
    if isinstance(first_content, str):
        # If the second chunk is also a string, then merge them naively
        if isinstance(second_content, str):
            return first_content + second_content
        # If the second chunk is a list, add the first chunk to the start of the list
        else:
            return_list: List[Union[str, Dict]] = [first_content]
            return return_list + second_content
    # If both are lists, merge them naively
    elif isinstance(second_content, List):
        return first_content + second_content
    # If the first content is a list, and the second content is a string
    else:
        # If the last element of the first content is a string
        # Add the second content to the last element
        if isinstance(first_content[-1], str):
            return first_content[:-1] + [first_content[-1] + second_content]
        else:
            # Otherwise, add the second content as a new element of the list
            return first_content + [second_content]


class BaseMessageChunk(BaseMessage):
    """A Message chunk, which can be concatenated with other Message chunks."""

    def _merge_kwargs_dict(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge additional_kwargs from another BaseMessageChunk into this one."""
        merged = left.copy()
        for k, v in right.items():
            if k not in merged:
                merged[k] = v
            elif type(merged[k]) != type(v):
                raise ValueError(
                    f'additional_kwargs["{k}"] already exists in this message,'
                    " but with a different type."
                )
            elif isinstance(merged[k], str):
                merged[k] += v
            elif isinstance(merged[k], dict):
                merged[k] = self._merge_kwargs_dict(merged[k], v)
            else:
                raise ValueError(
                    f"Additional kwargs key {k} already exists in this message."
                )
        return merged

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, BaseMessageChunk):
            # If both are (subclasses of) BaseMessageChunk,
            # concat into a single BaseMessageChunk

            if isinstance(self, ChatMessageChunk):
                return self.__class__(
                    role=self.role,
                    content=merge_content(self.content, other.content),
                    additional_kwargs=self._merge_kwargs_dict(
                        self.additional_kwargs, other.additional_kwargs
                    ),
                )
            return self.__class__(
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )
        else:
            raise TypeError(
                'unsupported operand type(s) for +: "'
                f"{self.__class__.__name__}"
                f'" and "{other.__class__.__name__}"'
            )


class HumanMessage(BaseMessage):
    """A Message from a human."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["human"] = "human"


HumanMessage.update_forward_refs()


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """A Human Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment] # noqa: E501


class AIMessage(BaseMessage):
    """A Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["ai"] = "ai"


AIMessage.update_forward_refs()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """A Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment] # noqa: E501

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, AIMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate AIMessageChunks with different example values."
                )

            return self.__class__(
                example=self.example,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class SystemMessage(BaseMessage):
    """A Message for priming AI behavior, usually passed in as the first of a sequence
    of input messages.
    """

    type: Literal["system"] = "system"


SystemMessage.update_forward_refs()


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """A System Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment] # noqa: E501


class FunctionMessage(BaseMessage):
    """A Message for passing the result of executing a function back to a model."""

    name: str
    """The name of the function that was executed."""

    type: Literal["function"] = "function"


FunctionMessage.update_forward_refs()


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """A Function Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["FunctionMessageChunk"] = "FunctionMessageChunk"  # type: ignore[assignment]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                raise ValueError(
                    "Cannot concatenate FunctionMessageChunks with different names."
                )

            return self.__class__(
                name=self.name,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class ToolMessage(BaseMessage):
    """A Message for passing the result of executing a tool back to a model."""

    tool_call_id: str
    """Tool call that this message is responding to."""

    type: Literal["tool"] = "tool"


ToolMessage.update_forward_refs()


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """A Tool Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                raise ValueError(
                    "Cannot concatenate ToolMessageChunks with different names."
                )

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class ChatMessage(BaseMessage):
    """A Message that can be assigned an arbitrary speaker (i.e. role)."""

    role: str
    """The speaker / role of the Message."""

    type: Literal["chat"] = "chat"


ChatMessage.update_forward_refs()


class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """A Chat Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ChatMessageChunk"] = "ChatMessageChunk"  # type: ignore

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ChatMessageChunk):
            if self.role != other.role:
                raise ValueError(
                    "Cannot concatenate ChatMessageChunks with different roles."
                )

            return self.__class__(
                role=self.role,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


AnyMessage = Union[
    AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage, ToolMessage
]


def _message_to_dict(message: BaseMessage) -> dict:
    return {"type": message.type, "data": message.dict()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> List[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [_message_to_dict(m) for m in messages]


def _message_from_dict(message: dict) -> BaseMessage:
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "chat":
        return ChatMessage(**message["data"])
    elif _type == "function":
        return FunctionMessage(**message["data"])
    elif _type == "tool":
        return ToolMessage(**message["data"])
    else:
        raise ValueError(f"Got unexpected message type: {_type}")


def messages_from_dict(messages: List[dict]) -> List[BaseMessage]:
    """Convert a sequence of messages from dicts to Message objects.

    Args:
        messages: Sequence of messages (as dicts) to convert.

    Returns:
        List of messages (BaseMessages).
    """
    return [_message_from_dict(m) for m in messages]
