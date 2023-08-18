from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.anthropic import _AnthropicCommon
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk


class ChatAnthropic(BaseChatModel, _AnthropicCommon):
    """`Anthropic` chat large language models.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            import anthropic
            from langchain.llms import Anthropic
            model = ChatAnthropic(model="<model_name>", anthropic_api_key="my-api-key")
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"anthropic_api_key": "ANTHROPIC_API_KEY"}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

    @property
    def lc_serializable(self) -> bool:
        return True

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
        messages = messages.copy()  # don't mutate the original list

        if not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if not isinstance(messages[-1], AIMessage):
            messages.append(AIMessage(content=""))
        text = self._convert_messages_to_text(messages)
        return (
            text.rstrip()
        )  # trim off the trailing ' ' that might come from the "Assistant: "

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params, **kwargs}
        if stop:
            params["stop_sequences"] = stop

        stream_resp = self.client.completions.create(**params, stream=True)
        for data in stream_resp:
            delta = data.completion
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params, **kwargs}
        if stop:
            params["stop_sequences"] = stop

        stream_resp = await self.async_client.completions.create(**params, stream=True)
        async for data in stream_resp:
            delta = data.completion
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                await run_manager.on_llm_new_token(delta)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            completion = ""
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                completion += chunk.text
        else:
            prompt = self._convert_messages_to_prompt(messages)
            params: Dict[str, Any] = {
                "prompt": prompt,
                **self._default_params,
                **kwargs,
            }
            if stop:
                params["stop_sequences"] = stop
            response = self.client.completions.create(**params)
            completion = response.completion
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            completion = ""
            async for chunk in self._astream(messages, stop, run_manager, **kwargs):
                completion += chunk.text
        else:
            prompt = self._convert_messages_to_prompt(messages)
            params: Dict[str, Any] = {
                "prompt": prompt,
                **self._default_params,
                **kwargs,
            }
            if stop:
                params["stop_sequences"] = stop
            response = await self.async_client.completions.create(**params)
            completion = response.completion
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        if not self.count_tokens:
            raise NameError("Please ensure the anthropic package is loaded")
        return self.count_tokens(text)
