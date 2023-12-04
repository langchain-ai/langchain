from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field

if TYPE_CHECKING:
    from langchain_core.prompts.chat import ChatPromptTemplate


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
        from langchain_core.prompts.chat import ChatPromptTemplate

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
        """Merge additional_kwargs from another BaseMessageChunk into this one,
        handling specific scenarios where a key exists in both dictionaries
        but has a value of None in 'left'. In such cases, the method uses the
        value from 'right' for that key in the merged dictionary.
        Example:
        If left = {"function_call": {"arguments": None}} and
        right = {"function_call": {"arguments": "{\n"}}
        then, after merging, for the key "function_call",
        the value from 'right' is used,
        resulting in merged = {"function_call": {"arguments": "{\n"}}.
        """
        merged = left.copy()
        for k, v in right.items():
            if k not in merged:
                merged[k] = v
            elif merged[k] is None and v:
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


def message_to_dict(message: BaseMessage) -> dict:
    return {"type": message.type, "data": message.dict()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> List[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [message_to_dict(m) for m in messages]
