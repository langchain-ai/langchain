"""Wrapper around Google VertexAI chat-based models."""
from typing import Dict, List, Optional, Tuple

from google.cloud.aiplatform.private_preview.language_models.language_models import (
    TextGenerationResponse,
    _MultiTurnChatSession,
)
from pydantic import root_validator

from langchain.chat_models.base import BaseChatModel
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms.vertex import _VertexAICommon
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)


class _ChatVertexAICommon(_VertexAICommon):
    """A base class for wrappers of Google Vertex AI chat LLMs.

    To use, you should have the
    ``google.cloud.aiplatform.private_preview.language_models`` python package
    installed.
    """

    model_name: str = "chat-bison-001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        try:
            from google.cloud.aiplatform.private_preview.language_models import (
                ChatModel,
            )

        except ImportError:
            raise ValueError("Could not import Vertex AI LLM python package. ")
        try:
            values["client"] = ChatModel.from_pretrained(values["model_name"])
        except AttributeError:
            raise ValueError("Could not initialize Vertex AI LLM.")

        return values

    def _response_to_chat_results(
        self, response: TextGenerationResponse, stop: Optional[List[str]]
    ) -> ChatResult:
        text = self._enforce_stop_words(response.text, stop)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class ChatVertexAI(_ChatVertexAICommon, BaseChatModel):
    """Wrapper around Vertex AI large language models."""

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        chat, prompt = self._start_chat(messages)
        response = chat.send_message(prompt)
        return self._response_to_chat_results(response, stop=stop)

    def _start_chat(
        self, messages: List[BaseMessage]
    ) -> Tuple[_MultiTurnChatSession, str]:
        """Start a chat.

        Args:
            messages: a list of BaseMessage.
        Returns:
            a tuple that has a Vertex AI chat model initializes, and a prompt to send to the model.

        Currently it expects either one HumanMessage, or two message (SystemMessage and HumanMessage).
        If two messages are provided, the first one would be use for context.
        """
        if len(messages) == 1:
            message = messages[0]
            if not isinstance(message, HumanMessage):
                raise ValueError("Message should be from human if it's the first one!")
            context, prompt = None, message.content
        elif len(messages) == 2:
            first_message, second_message = messages[0], messages[1]
            if not isinstance(first_message, SystemMessage):
                raise ValueError(
                    "The first message should be a system one if there're two of them."
                )
            if not isinstance(second_message, HumanMessage):
                raise ValueError("The second message should from human!")
            context, prompt = first_message.content, second_message.content
        else:
            raise ValueError("Chat model expects only one or two messages!")
        chat = self.client.start_chat(context=context, **self._default_params)
        return chat, prompt

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        raise NotImplementedError(
            """Vertex AI doesn't support async requests at the moment."""
        )


class MultiTurnChatVertexAI(_ChatVertexAICommon, BaseChatModel):
    """Wrapper around Vertex AI large language models."""

    chat: Optional[_MultiTurnChatSession] = None

    def clear_chat(self) -> None:
        self.chat = None

    def start_chat(self, message: Optional[SystemMessage] = None) -> None:
        if self.chat:
            raise ValueError("Chat has already been started! Please, clear it first.")
        if message and not isinstance(message, SystemMessage):
            raise ValueError("Context should be a system message")
        context = message.content if message else None
        self.chat = self.client.start_chat(context=context, **self._default_params)

    @property
    def history(self) -> List[Tuple[str]]:
        """Chat history."""
        if self.chat:
            return self.chat._history
        return []

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        if len(messages) != 1:
            raise ValueError(
                "You should send exactly one message to the chat each turn."
            )
        if not self.chat:
            raise ValueError("You should start_chat first!")
        response = self.chat.send_message(messages[0].content)
        return self._response_to_chat_results(response, stop=stop)

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        raise NotImplementedError(
            """Vertex AI doesn't support async requests at the moment."""
        )
