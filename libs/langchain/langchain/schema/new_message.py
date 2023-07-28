T = TypeVar("T")


class PromptInterface(Generic[T], ABC):
    @abstractmethod
    def format(self, **kwargs: Any) -> T:
        ...

    def __str__(self):
        raise NotImplementedError


class Prompt(PromptInterface[str]):
    def __init__(
        self,
        content: Optional[str] = None,
        input_variables: Optional[Mapping[str, Any]] = None,
        formatter: Literal["f-string", "jinja2"] = "f-string",
    ) -> None:
        self.content = content
        self.input_variables = input_variables or infer_inputs(content, formatter)
        self.formatter = formatter

    def format(self, **kwargs: Any) -> str:
        if not self.input_variables:
            return self.content
        elif self.formatter == "f-string":
            return self.content.format(**kwargs)
        elif self.formatter == "jinja":
            return
        else:
            raise ValueError

    def __str__(self):
        return self.content


class Message(PromptInterface["Message"]):
    def __init__(
        self,
        prompt: Prompt,
        role: Optional[str] = None,
        type: Optional[Literal["human", "ai", "system", "function"]] = None,
        additional_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.role = role
        self.type = type
        self.additional_data = additional_data

    @classmethod
    def from_content(
        cls,
        content: Optional[str] = None,
        input_variables: Optional[Mapping[str, Any]] = None,
        formatter: Literal["f-string", "jinja2"] = "f-string",
        **kwargs: Any,
    ) -> Message:
        prompt = Prompt(
            content=content, input_variables=input_variables, formatter=formatter
        )
        return cls(prompt, **kwargs)

    def format(self, **kwargs: Any) -> Message:
        content = self.prompt.format(**kwargs)
        return self.from_content(
            content=content,
            role=self.role,
            type=self.type,
            additional_data=self.additional_data,
        )

    def __add__(self, other) -> MessageSequence:
        return MessageSequence((self, other))

    def __str__(self):
        TYPE_TO_NAME = {
            "human": "Human",
            "ai": "AI",
            "function": "Function",
            "system": "System",
        }
        name = self.role if self.role else TYPE_TO_NAME[self.type]
        return f"{name}: {str(self.prompt)}"


class MessageSequence(PromptInterface[List[Message]]):
    def __init__(self, messages: Sequence[Message]):
        self.messages = messages

    def format(self, **kwargs) -> List[Message]:
        return [m.format(**kwargs) for m in self.messages]

    def __add__(self, other) -> MessageSequence:
        return MessageSequence(list(self.messages) + list(other.messages))

    def __getitem__(self, item):
        return self.messages.__getitem__(item)

    def __iter__(self):
        return self.messages.__iter__

    def __str__(self):
        return "\n".join(str(m) for m in self.messages)
