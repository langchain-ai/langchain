from __future__ import annotations

from abc import ABC, abstractmethod
from string import Formatter
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from langchain.utils import formatter

T = TypeVar("T")


def _default_repr(obj: Any) -> str:
    params = []
    for k, v in getattr(obj, "__dict__", {}).items():
        if hasattr(v, "__dict__"):
            v = _default_repr(v)
        elif isinstance(v, str):
            v = f'"{v}"'
        params.append(f"{k}={v}")
    params_str = ", ".join(params)
    return f"{obj.__class__.__name__}({params_str})"


class PromptInterface(Generic[T], ABC):
    @abstractmethod
    def format(self, **kwargs: Any) -> T:
        ...

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return _default_repr(self)


FormatterType = Literal["f-string", "jinja2"]


def infer_inputs(content: str, formatter_: FormatterType) -> Tuple[str, ...]:
    if formatter_ == "f-string":
        return tuple({v for _, v, _, _ in Formatter().parse(content) if v is not None})
    elif formatter_ == "jinja2":
        raise NotImplementedError
    else:
        raise ValueError


PromptLike = Union["Prompt", str]


class Prompt(PromptInterface[str]):
    def __init__(
        self,
        content: str,
        *,
        input_variables: Sequence[str] = (),
        formatter_: FormatterType = "f-string",
    ) -> None:
        self.content = content
        self.input_variables = input_variables or infer_inputs(content, formatter_)
        self.formatter_ = formatter_

    def format(self, **kwargs: Any) -> str:
        if not self.input_variables:
            return self.content
        elif self.formatter_ == "f-string":
            relevant_kwargs = {
                k: v for k, v in kwargs.items() if k in self.input_variables
            }
            filler_kwargs = {
                iv: "{" + iv + "}"
                for iv in self.input_variables
                if iv not in relevant_kwargs
            }
            return formatter.format(self.content, **relevant_kwargs, **filler_kwargs)
        elif self.formatter_ == "jinja2":
            raise NotImplementedError
        else:
            raise ValueError

    def __str__(self) -> str:
        return self.content

    def __add__(self, other: PromptLike) -> Prompt:
        if isinstance(other, Prompt):
            if self.formatter_ != other.formatter_:
                raise ValueError
            content = self.content + other.content
            input_variables = tuple(
                set(self.input_variables) | set(other.input_variables)
            )
            return Prompt(
                content, input_variables=input_variables, formatter_=self.formatter_
            )
        elif isinstance(other, str):
            content = self.content + other
            return Prompt(content, formatter_=self.formatter_)
        else:
            raise ValueError

    def __radd__(self, other: PromptLike) -> Prompt:
        if isinstance(other, str):
            other = Prompt(other, formatter_=self.formatter_)
        elif isinstance(other, Prompt):
            pass
        else:
            raise ValueError
        return other.__add__(self)


MessageLike = Union["Message", Tuple[str, PromptLike], Mapping[str, PromptLike]]
MessageAddable = Union[MessageLike, "MessageSequence"]
MessageType = Literal["human", "ai", "system", "function"]
MSG_TYPES = {"human", "ai", "function", "system"}
MSG_TYPE_TO_NAME = {
    "human": "Human",
    "ai": "AI",
    "function": "Function",
    "system": "System",
}


class Message(PromptInterface["Message"]):
    def __init__(
        self,
        prompt: PromptLike,
        *,
        role: Optional[str] = None,
        type: Optional[MessageType] = None,
        additional_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
        self.role = role
        self.type = type if role or type else "human"
        self.additional_data = additional_data

    def format(self, **kwargs: Any) -> Message:
        prompt = Prompt(self.prompt.format(**kwargs))
        return self.__class__(
            prompt, role=self.role, type=self.type, additional_data=self.additional_data
        )

    @classmethod
    def coerce(cls, message_like: MessageLike) -> Message:
        if isinstance(message_like, Message):
            return message_like
        elif isinstance(message_like, (tuple, Mapping)):
            message_tuple = (
                message_like
                if isinstance(message_like, tuple)
                else list(message_like.items())[0]
            )
            role_or_type, prompt_like = message_tuple
            kwargs: Dict = {}
            if role_or_type in MSG_TYPES:
                kwargs["type"] = role_or_type
            else:
                kwargs["role"] = role_or_type
            return cls(prompt_like, **kwargs)
        else:
            raise ValueError

    def __add__(self, other: MessageAddable) -> MessageSequence:
        if isinstance(other, Message):
            return MessageSequence((self, other))
        elif isinstance(other, MessageSequence):
            return MessageSequence((self, *other.messages))
        elif isinstance(other, (tuple, Mapping)):
            message = self.coerce(other)
            return MessageSequence((self, message))
        else:
            raise ValueError

    def __radd__(self, other: MessageAddable) -> MessageSequence:
        if isinstance(other, Message):
            return MessageSequence((other, self))
        elif isinstance(other, MessageSequence):
            return MessageSequence((*other.messages, self))
        elif isinstance(other, (tuple, Mapping)):
            message = self.coerce(other)
            return MessageSequence((message, self))
        else:
            raise ValueError

    def __str__(self) -> str:
        name = self.role if self.role else MSG_TYPE_TO_NAME[self.type]
        return f"{name}: {str(self.prompt)}"


class MessageSequence(PromptInterface[List[Message]]):
    def __init__(self, messages: Sequence[MessageLike]) -> None:
        self.messages = tuple(Message.coerce(m) for m in messages)

    def format(self, **kwargs: Any) -> List[Message]:
        return [m.format(**kwargs) for m in self.messages]

    @classmethod
    def coerce(
        cls, message_seq_like: Union[MessageAddable, Sequence[MessageLike]]
    ) -> MessageSequence:
        if isinstance(message_seq_like, (Message, Mapping)):
            return MessageSequence((message_seq_like,))
        elif isinstance(message_seq_like, Sequence):
            if (
                isinstance(message_seq_like, tuple)
                and len(message_seq_like) == 2
                and isinstance(message_seq_like, str)
            ):
                return MessageSequence((message_seq_like,))
            else:
                return MessageSequence(message_seq_like)
        elif isinstance(message_seq_like, MessageSequence):
            return message_seq_like
        else:
            raise ValueError

    def __add__(
        self, other: Union[MessageAddable, Sequence[MessageLike]]
    ) -> MessageSequence:
        other_seq = self.coerce(other)
        return self.__class__((*self.messages, *other_seq.messages))

    def __radd__(
        self, other: Union[MessageAddable, Sequence[MessageLike]]
    ) -> MessageSequence:
        other_seq = self.coerce(other)
        return other_seq.__add__(self)

    def __getitem__(self, index: int) -> Message:
        return self.messages.__getitem__(index)

    def __iter__(self) -> Iterator[Message]:
        return self.messages.__iter__()

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.messages)
