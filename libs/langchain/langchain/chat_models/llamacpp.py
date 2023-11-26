from typing import Any, List, Optional, Iterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.llamacpp import LlamaCpp
from langchain_core.outputs import ChatGenerationChunk


class ChatLlamacpp(BaseChatModel, LlamaCpp):
    """LLamacpp locally runs large language models.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatLlamacpp
            llamacpp = ChatLlamacpp(model_path="./models/llama2")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "llamacpp-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    def _format_message_as_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"[INST] {message.content} [/INST]"
        elif isinstance(message, AIMessage):
            message_text = f"{message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"<<SYS>> {message.content} <</SYS>>"
        else:
            raise ValueError(f"Got unknown type {message}, type {type(message)}")
        return message_text

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            [self._format_message_as_text(message) for message in messages]
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to LLamacpp's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

               response = llamacpp([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """

        prompt = self._format_messages_as_text(messages)
        response = LlamaCpp._call(self,
            prompt=prompt, run_manager=run_manager, stop=stop, **kwargs
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=response),
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        return LlamaCpp._stream(self, prompt, stop, run_manager, **kwargs)
