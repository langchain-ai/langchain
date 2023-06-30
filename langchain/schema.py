"""Common schema objects."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
from uuid import UUID

from pydantic import BaseModel, Field, root_validator

from langchain.load.serializable import Serializable

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
        Callbacks,
    )

RUN_KEY = "__run"


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


@dataclass
class AgentAction:
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class Generation(Serializable):
    """Output of a single generation."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider"""
    """May include things like reason for finishing (e.g. in OpenAI)"""
    # TODO: add log probs

    @property
    def lc_serializable(self) -> bool:
        """This class is LangChain serializable."""
        return True


class BaseMessage(Serializable):
    """Message object."""

    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""

    @property
    def lc_serializable(self) -> bool:
        """This class is LangChain serializable."""
        return True


class HumanMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "human"


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    example: bool = False

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


class FunctionMessage(BaseMessage):
    name: str

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "function"


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
    """Convert messages to dict.

    Args:
        messages: List of messages to convert.

    Returns:
        List of dicts.
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
    else:
        raise ValueError(f"Got unexpected type: {_type}")


def messages_from_dict(messages: List[dict]) -> List[BaseMessage]:
    """Convert messages from dict.

    Args:
        messages: List of messages (dicts) to convert.

    Returns:
        List of messages (BaseMessages).
    """
    return [_message_from_dict(m) for m in messages]


class ChatGeneration(Generation):
    """Output of a single generation."""

    text = ""
    message: BaseMessage

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["text"] = values["message"].content
        return values


class RunInfo(BaseModel):
    """Class that contains all relevant metadata for a Run."""

    run_id: UUID


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
    run: Optional[List[RunInfo]] = None
    """Run metadata."""

    def flatten(self) -> List[LLMResult]:
        """Flatten generations into a single list."""
        llm_results = []
        for i, gen_list in enumerate(self.generations):
            # Avoid double counting tokens in OpenAICallback
            if i == 0:
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=self.llm_output,
                    )
                )
            else:
                if self.llm_output is not None:
                    llm_output = self.llm_output.copy()
                    llm_output["token_usage"] = dict()
                else:
                    llm_output = None
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=llm_output,
                    )
                )
        return llm_results

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LLMResult):
            return NotImplemented
        return (
            self.generations == other.generations
            and self.llm_output == other.llm_output
        )


class PromptValue(Serializable, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""


class BaseMemory(Serializable, ABC):
    """Base interface for memory in chains."""

    class Config:
        """Configuration for this pydantic object."""

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


class BaseChatMessageHistory(ABC):
    """Base interface for chat message history
    See `ChatMessageHistory` for default implementation.
    """

    """
    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)
               
               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]

    def add_user_message(self, message: str) -> None:
        """Add a user message to the store"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the store"""
        self.add_message(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""


class Document(Serializable):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)


class BaseRetriever(ABC):
    """Base interface for a retriever."""

    _new_arg_supported: bool = False
    _expects_other_args: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Version upgrade for old retrievers that implemented the public
        # methods directly.
        if cls.get_relevant_documents != BaseRetriever.get_relevant_documents:
            warnings.warn(
                "Retrievers must implement abstract `_get_relevant_documents` method"
                " instead of `get_relevant_documents`",
                DeprecationWarning,
            )
            swap = cls.get_relevant_documents
            cls.get_relevant_documents = (  # type: ignore[assignment]
                BaseRetriever.get_relevant_documents
            )
            cls._get_relevant_documents = swap  # type: ignore[assignment]
        if (
            hasattr(cls, "aget_relevant_documents")
            and cls.aget_relevant_documents != BaseRetriever.aget_relevant_documents
        ):
            warnings.warn(
                "Retrievers must implement abstract `_aget_relevant_documents` method"
                " instead of `aget_relevant_documents`",
                DeprecationWarning,
            )
            aswap = cls.aget_relevant_documents
            cls.aget_relevant_documents = (  # type: ignore[assignment]
                BaseRetriever.aget_relevant_documents
            )
            cls._aget_relevant_documents = aswap  # type: ignore[assignment]
        parameters = signature(cls._get_relevant_documents).parameters
        cls._new_arg_supported = parameters.get("run_manager") is not None
        # If a V1 retriever broke the interface and expects additional arguments
        cls._expects_other_args = (not cls._new_arg_supported) and len(parameters) > 2

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

    @abstractmethod
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

    def get_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        from langchain.callbacks.manager import CallbackManager

        callback_manager = CallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_manager = callback_manager.on_retriever_start(
            query,
            **kwargs,
        )
        try:
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    query, run_manager=run_manager, **kwargs
                )
            elif self._expects_other_args:
                result = self._get_relevant_documents(query, **kwargs)
            else:
                result = self._get_relevant_documents(query)  # type: ignore[call-arg]
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        from langchain.callbacks.manager import AsyncCallbackManager

        callback_manager = AsyncCallbackManager.configure(
            callbacks, None, verbose=kwargs.get("verbose", False)
        )
        run_manager = await callback_manager.on_retriever_start(
            query,
            **kwargs,
        )
        try:
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
                    query, run_manager=run_manager, **kwargs
                )
            elif self._expects_other_args:
                result = await self._aget_relevant_documents(query, **kwargs)
            else:
                result = await self._aget_relevant_documents(
                    query,  # type: ignore[call-arg]
                )
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result


# For backwards compatibility


Memory = BaseMemory

T = TypeVar("T")


class BaseLLMOutputParser(Serializable, ABC, Generic[T]):
    @abstractmethod
    def parse_result(self, result: List[Generation]) -> T:
        """Parse LLM Result."""


class BaseOutputParser(BaseLLMOutputParser, ABC, Generic[T]):
    """Class to parse the output of an LLM call.

    Output parsers help structure language model responses.
    """

    def parse_result(self, result: List[Generation]) -> T:
        return self.parse(result[0].text)

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse the output of an LLM call.

        A method which takes in a string (assumed output of a language model )
        and parses it into some structure.

        Args:
            text: output of language model

        Returns:
            structured output
        """

    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        """Optional method to parse the output of an LLM call with a prompt.

        The prompt is largely provided in the event the OutputParser wants
        to retry or fix the output in some way, and needs information from
        the prompt to do so.

        Args:
            completion: output of language model
            prompt: prompt value

        Returns:
            structured output
        """
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the type key."""
        raise NotImplementedError(
            f"_type property is not implemented in class {self.__class__.__name__}."
            " This is required for serialization."
        )

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict()
        output_parser_dict["_type"] = self._type
        return output_parser_dict


class NoOpOutputParser(BaseOutputParser[str]):
    """Output parser that just returns the text as is."""

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def _type(self) -> str:
        return "default"

    def parse(self, text: str) -> str:
        return text


class OutputParserException(ValueError):
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. OutputParserExceptions will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.
    """

    def __init__(
        self,
        error: Any,
        observation: str | None = None,
        llm_output: str | None = None,
        send_to_llm: bool = False,
    ):
        super(OutputParserException, self).__init__(error)
        if send_to_llm:
            if observation is None or llm_output is None:
                raise ValueError(
                    "Arguments 'observation' & 'llm_output'"
                    " are required if 'send_to_llm' is True"
                )
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm


class BaseDocumentTransformer(ABC):
    """Base interface for transforming documents."""

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents."""

    @abstractmethod
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents."""
