import asyncio
from functools import partial
from typing import Any, List, Optional, Tuple, cast

from ai21.models import ChatMessage, Penalty, RoleType
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_ai21.ai21_base import AI21Base


def _get_system_message_from_message(message: BaseMessage) -> str:
    if not isinstance(message.content, str):
        raise ValueError(
            f"System Message must be of type str. Got {type(message.content)}"
        )

    return message.content


def _convert_messages_to_ai21_messages(
    messages: List[BaseMessage],
) -> Tuple[Optional[str], List[ChatMessage]]:
    system_message = None
    converted_messages: List[ChatMessage] = []

    for i, message in enumerate(messages):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            else:
                system_message = _get_system_message_from_message(message)
        else:
            converted_message = _convert_message_to_ai21_message(message)
            converted_messages.append(converted_message)

    return system_message, converted_messages


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

    model: str
    """Model type you wish to interact with. 
        You can view the options at https://github.com/AI21Labs/ai21-python?tab=readme-ov-file#model-types"""
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
    """A penalty applied to tokens based on their frequency 
    in the generated responses."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ai21"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, ai21_messages = _convert_messages_to_ai21_messages(messages)

        response = self.client.chat.create(
            model=self.model,
            messages=ai21_messages,
            system=system or "",
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
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), messages, stop, run_manager
        )
