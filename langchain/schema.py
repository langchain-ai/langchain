"""Common schema objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional

from pydantic import BaseModel, Extra, Field, root_validator


def get_buffer_string(
    messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Get buffer string of messages."""
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        string_messages.append(f"{role}: {m.content}")
    return "\n".join(string_messages)


class AgentAction(NamedTuple):
    """Agent's action to take."""

    tool: str
    tool_input: str
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class Generation(BaseModel):
    """Output of a single generation."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider"""
    """May include things like reason for finishing (e.g. in OpenAI)"""
    # TODO: add log probs


class BaseMessage(BaseModel):
    """Message object."""

    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""


class HumanMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "human"


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ai"


class SystemMessage(BaseMessage):
    """Type of message that is a system message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "system"


class ChatMessage(BaseMessage):
    """Type of message with arbitrary speaker."""

    role: str

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "chat"


def _message_to_dict(message: BaseMessage) -> dict:
    return {"type": message.type, "data": message.dict()}


def messages_to_dict(messages: List[BaseMessage]) -> List[dict]:
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
    else:
        raise ValueError(f"Got unexpected type: {_type}")


def messages_from_dict(messages: List[dict]) -> List[BaseMessage]:
    return [_message_from_dict(m) for m in messages]


class ChatGeneration(Generation):
    """Output of a single generation."""

    text = ""
    message: BaseMessage

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["text"] = values["message"].content
        return values


class ChatResult(BaseModel):
    """Class that contains all relevant information for a Chat Result."""

    generations: List[ChatGeneration]
    """List of the things generated."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class LLMResult(BaseModel):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Generation]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""


class BaseLanguageModel(BaseModel, ABC):
    @abstractmethod
    def generate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    @abstractmethod
    async def agenerate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        # TODO: this method may not be exact.
        # TODO: this method may differ based on model (eg codex).
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install transformers`."
            )
        # create a GPT-3 tokenizer instance
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # tokenize the text using the GPT-3 tokenizer
        tokenized_text = tokenizer.tokenize(text)

        # calculate the number of tokens in the tokenized text
        return len(tokenized_text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the message."""
        return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])


class BaseMemory(BaseModel, ABC):
    """Base interface for memory in chains."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    lookup_str: str = ""
    lookup_index = 0
    metadata: dict = Field(default_factory=dict)

    @property
    def paragraphs(self) -> List[str]:
        """Paragraphs of the page."""
        return self.page_content.split("\n\n")

    @property
    def summary(self) -> str:
        """Summary of the page (the first paragraph)."""
        return self.paragraphs[0]

    def lookup(self, string: str) -> str:
        """Lookup a term in the page, imitating cmd-F functionality."""
        if string.lower() != self.lookup_str:
            self.lookup_str = string.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self.paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"


class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_texts(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant tests for

        Returns:
            List of relevant documents
        """


# For backwards compatibility


Memory = BaseMemory


class BaseOutputParser(BaseModel, ABC):
    """Class to parse the output of an LLM call."""

    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call."""

    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict()
        output_parser_dict["_type"] = self._type
        return output_parser_dict


class OutputParserException(Exception):
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. OutputParserExceptions will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.
    """

    pass
