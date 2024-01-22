from typing import Any, AsyncIterator, Iterator, List, Optional, cast

from ai21.models import ChatMessage, RoleType, Penalty

from langchain_ai21.ai21_base import AI21Base
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _convert_message_to_ai21_message(
    message: BaseMessage,
) -> ChatMessage:
    content = cast(str, message.content)

    role = None

    if isinstance(message, HumanMessage):
        role = RoleType.USER
    elif isinstance(message, AIMessage):
        role = RoleType.ASSISTANT

    if not role:
        raise ValueError(
            f"Could not resolve role type from message {message}. "
            f"Only support {HumanMessage.__name__} and {AIMessage.__name__}."
        )

    return ChatMessage(role=role, text=content)


def _pop_system_messages(messages: List[BaseMessage]) -> List[SystemMessage]:
    system_message_indexes = [
        i for i, message in enumerate(messages) if isinstance(message, SystemMessage)
    ]

    return [cast(SystemMessage, messages.pop(i)) for i in system_message_indexes]


class ChatAI21(BaseChatModel, AI21Base):
    """ChatAI21 chat model.

    Example:
        .. code-block:: python

            from langchain_ai21 import ChatAI21


            model = ChatAI21()
    """

    model: str = "j2-ultra"
    num_results: int = 1
    """The number of responses to generate for a given prompt."""

    max_tokens: int = 16
    """The maximum number of tokens to generate for each response."""

    min_tokens: int = 0
    """The minimum number of tokens to generate for each response."""

    temperature: float = 0.7
    """A value controlling the "creativity" of the model's responses."""

    top_p: float = 1
    """A value controlling the diversity of the model's responses."""

    top_k_return: int = 0
    """The number of top-scoring tokens to consider for each generation step."""

    frequency_penalty: Optional[Penalty] = None
    """A penalty applied to tokens that are frequently generated."""

    presence_penalty: Optional[Penalty] = None
    """ A penalty applied to tokens that are already present in the prompt."""

    count_penalty: Optional[Penalty] = None
    """A penalty applied to tokens based on their frequency in the generated responses."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ai21"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=BaseMessageChunk(content="Yield chunks", type="ai"),
        )
        yield ChatGenerationChunk(
            message=BaseMessageChunk(content=" like this!", type="ai"),
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = messages.copy()
        system_messages = _pop_system_messages(messages)
        last_system_message_str = system_messages[-1].content if system_messages else ""
        ai21_messages = [
            _convert_message_to_ai21_message(message) for message in messages
        ]

        response = self.client.chat.create(
            model=self.model,
            messages=ai21_messages,
            system=last_system_message_str,
            num_results=self.num_results,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            top_p=self.top_p,
            top_k_return=self.top_k_return,
            stop_sequences=stop,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            count_penalty=self.count_penalty,
            **kwargs,
        )

        outputs = response.outputs
        message = AIMessage(content=outputs[0].text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError
