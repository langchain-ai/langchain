class Prompt:
    def __init__(
        self,
        content: Optional[str] = None,
        input_variables: Optional[Mapping[str, Any]] = None,
        formatter: Literal["f-string", "jinja2"] = "f-string",
    ) -> None:
        self.content = content
        self.input_variables = input_variables or infer_inputs(content, formatter)
        self.formatter = formatter

    def format(self: T, **kwargs: Any) -> T:
        if not self.input_variables:
            return self
        elif self.formatter == "f-string":
            return self.content.format(**kwargs)
        elif self.formatter == "jinja":
            return
        else:
            raise ValueError


@dataclasses.dataclass(frozen=True)
class Message(Prompt):
    def __init__(
        self,
        role: Optional[str] = None,
        type: Optional[Literal["human", "ai", "system", "function"]] = None,
        additional_data: Optional[Mapping[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.type = type
        self.additional_data = additional_data

    def __add__(self, other) -> MessageCollection:
        return MessageCollection(self, other)


@dataclasses.dataclass(frozen=True)
class MessageCollection(Prompt):
    messages: Sequence[Message]
