"""Common schema objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional

from pydantic import BaseModel, Field, root_validator


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


class HumanMessage(BaseMessage):
    """Type of message that is spoken by the human."""


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""


class SystemMessage(BaseMessage):
    """Type of message that is a system message."""


class ChatMessage(BaseMessage):
    """Type of message with arbitrary speaker."""

    role: str


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
