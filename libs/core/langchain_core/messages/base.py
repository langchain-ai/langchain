from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.utils import get_bolded_text
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessage(Serializable):
    """Base abstract Message class.

    Messages are the inputs and outputs of ChatModels.
    """

    content: Union[str, List[Union[str, Dict]]]
    """The string contents of the message."""

    additional_kwargs: dict = Field(default_factory=dict)
    """Any additional information."""

    type: str

    class Config:
        extra = Extra.allow

    def __init__(
        self, content: Union[str, List[Union[str, Dict]]], **kwargs: Any
    ) -> None:
        """Pass in content as positional arg."""
        return super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> ChatPromptTemplate:
        from langchain_core.prompts.chat import ChatPromptTemplate

        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other

    def pretty_repr(self, html: bool = False) -> str:
        title = get_msg_title_repr(self.type.title() + " Message", bold=html)
        # TODO: handle non-string content.
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


def merge_content(
    first_content: Union[str, List[Union[str, Dict]]],
    second_content: Union[str, List[Union[str, Dict]]],
) -> Union[str, List[Union[str, Dict]]]:
    """Merge two message contents.

    Args:
        first_content: The first content.
        second_content: The second content.

    Returns:
        The merged content.
    """
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
    """Message chunk, which can be concatenated with other Message chunks."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

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
            elif v is None:
                continue
            elif merged[k] == v:
                continue
            elif type(merged[k]) != type(v):
                raise TypeError(
                    f'additional_kwargs["{k}"] already exists in this message,'
                    " but with a different type."
                )
            elif isinstance(merged[k], str):
                merged[k] += v
            elif isinstance(merged[k], dict):
                merged[k] = self._merge_kwargs_dict(merged[k], v)
            elif isinstance(merged[k], list):
                merged[k] = merged[k].copy()
                for i, e in enumerate(v):
                    if isinstance(e, dict) and isinstance(e.get("index"), int):
                        i = e["index"]
                    if i < len(merged[k]):
                        merged[k][i] = self._merge_kwargs_dict(merged[k][i], e)
                    else:
                        merged[k] = merged[k] + [e]
            else:
                raise TypeError(
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
    """Convert a Message to a dictionary.

    Args:
        message: Message to convert.

    Returns:
        Message as a dict.
    """
    return {"type": message.type, "data": message.dict()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> List[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [message_to_dict(m) for m in messages]


def get_msg_title_repr(title: str, *, bold: bool = False) -> str:
    """Get a title representation for a message.

    Args:
        title: The title.
        bold: Whether to bold the title.

    Returns:
        The title representation.
    """
    padded = " " + title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    if bold:
        padded = get_bolded_text(padded)
    return f"{sep}{padded}{second_sep}"
