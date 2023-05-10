from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.anthropic import _AnthropicCommon
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)


class ChatAnthropic(BaseChatModel, _AnthropicCommon):
    r"""Wrapper around Anthropic's large language model.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python
            import anthropic
            from langchain.llms import Anthropic
            model = ChatAnthropic(model="<model_name>", anthropic_api_key="my-api-key")
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

    def _convert_one_message_to_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"{self.HUMAN_PROMPT} {message.content}"
        elif isinstance(message, AIMessage):
            message_text = f"{self.AI_PROMPT} {message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"{self.HUMAN_PROMPT} <admin>{message.content}</admin>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Format a list of strings into a single string with necessary newlines.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary newlines.
        """
        return "".join(
            self._convert_one_message_to_text(message) for message in messages
        )

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a full prompt for the Anthropic model

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
        if not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if not isinstance(messages[-1], AIMessage):
            messages.append(AIMessage(content=""))
        text = self._convert_messages_to_text(messages)
        return (
            text.rstrip()
        )  # trim off the trailing ' ' that might come from the "Assistant: "

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params}
        if stop:
            params["stop_sequences"] = stop

        if self.streaming:
            completion = ""
            stream_resp = self.client.completion_stream(**params)
            for data in stream_resp:
                delta = data["completion"][len(completion) :]
                completion = data["completion"]
                if run_manager:
                    run_manager.on_llm_new_token(
                        delta,
                    )
        else:
            response = self.client.completion(**params)
            completion = response["completion"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params}
        if stop:
            params["stop_sequences"] = stop

        if self.streaming:
            completion = ""
            stream_resp = await self.client.acompletion_stream(**params)
            async for data in stream_resp:
                delta = data["completion"][len(completion) :]
                completion = data["completion"]
                if run_manager:
                    await run_manager.on_llm_new_token(
                        delta,
                    )
        else:
            response = await self.client.acompletion(**params)
            completion = response["completion"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])
