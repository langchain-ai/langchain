class PromptInterface(ABC):

    @abstractmethod
    def format(self, **kwargs: Any) -> Any:

class Prompt(PromptInterface):
    def __init__(
        self,
        content: Optional[str] = None,
        input_variables: Optional[Mapping[str, Any]] = None,
        formatter: Literal["f-string", "jinja2"] = "f-string",
    ) -> None:
        self.content = content
        self.input_variables = input_variables or infer_inputs(content, formatter)
        self.formatter = formatter

    def format(self, **kwargs: Any) -> Any:
        if not self.input_variables:
            return self.content
        elif self.formatter == "f-string":
            return self.content.format(**kwargs)
        elif self.formatter == "jinja":
            return
        else:
            raise ValueError


class Message(Prompt):
    def __init__(
        self,
        role: Optional[str] = None,
        type: Optional[Literal["human", "ai", "system", "function"]] = None,
        additional_data: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.type = type
        self.additional_data = additional_data

    def __add__(self, other) -> MessageCollection:
        return MessageCollection((self, other))


class MessageSequence(PromptInterface):
    def __init__(self, messages: Sequence[Message]):
        self.messages = messages

    def format(self, **kwargs) -> List[Message]:
        return [m.format(**kwargs) for m in self.messages]

    def __add__(self, other) -> MessageCollection:
        return MessageCollection(list(self.messages) + list(other.messages))


