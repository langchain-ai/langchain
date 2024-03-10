from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_community.llms.friendli import BaseFriendli


def get_role(message: BaseMessage) -> str:
    """Get role of the message.

    Args:
        message (BaseMessage): The message object.

    Raises:
        ValueError: Raised when the message is of an unknown type.

    Returns:
        str: The role of the message.
    """
    if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    raise ValueError(f"Got unknown type {message}")


def get_chat_request(messages: List[BaseMessage]) -> Dict[str, Any]:
    """Get a request of the Friendli chat API.

    Args:
        messages (List[BaseMessage]): Messages comprising the conversation so far.

    Returns:
        Dict[str, Any]: The request for the Friendli chat API.
    """
    return {
        "messages": [
            {"role": get_role(message), "content": message.content}
            for message in messages
        ]
    }


class ChatFriendli(BaseChatModel, BaseFriendli):
    """Friendli LLM for chat.

    ``friendli-client`` package should be installed with `pip install friendli-client`.
    You must set ``FRIENDLI_TOKEN`` environment variable or provide the value of your
    personal access token for the ``friendli_token`` argument.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import FriendliChat

            chat = Friendli(
                model="llama-2-13b-chat", friendli_token="YOUR FRIENDLI TOKEN"
            )
            chat.invoke("What is generative AI?")
    """

    model: str = "llama-2-13b-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"friendli_token": "FRIENDLI_TOKEN"}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Friendli completions API."""
        return {
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model, **self._default_params}

    @property
    def _llm_type(self) -> str:
        return "friendli-chat"

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params = self._default_params
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            params["stop"] = self.stop
        else:
            params["stop"] = stop
        return {**params, **kwargs}

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._get_invocation_params(stop=stop, **kwargs)
        stream = self.client.chat.completions.create(
            **get_chat_request(messages), stream=True, model=self.model, **params
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
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
        params = self._get_invocation_params(stop=stop, **kwargs)
        stream = await self.async_client.chat.completions.create(
            **get_chat_request(messages), stream=True, model=self.model, **params
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
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
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        params = self._get_invocation_params(stop=stop, **kwargs)
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": get_role(message),
                    "content": message.content,
                }
                for message in messages
            ],
            stream=False,
            model=self.model,
            **params,
        )

        message = AIMessage(content=response.choices[0].message.content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        params = self._get_invocation_params(stop=stop, **kwargs)
        response = await self.async_client.chat.completions.create(
            messages=[
                {
                    "role": get_role(message),
                    "content": message.content,
                }
                for message in messages
            ],
            stream=False,
            model=self.model,
            **params,
        )

        message = AIMessage(content=response.choices[0].message.content)
        return ChatResult(generations=[ChatGeneration(message=message)])
