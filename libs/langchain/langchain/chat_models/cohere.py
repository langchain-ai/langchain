from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import (
    BaseChatModel,
    _agenerate_from_stream,
    _generate_from_stream,
)
from langchain.llms.cohere import BaseCohere
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGeneration, ChatGenerationChunk, ChatResult


def get_role(message: BaseMessage) -> str:
    """Get the role of the message.

    Args:
        message: The message.

    Returns:
        The role of the message.

    Raises:
        ValueError: If the message is of an unknown type.
    """
    if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
        return "User"
    elif isinstance(message, AIMessage):
        return "Chatbot"
    elif isinstance(message, SystemMessage):
        return "System"
    else:
        raise ValueError(f"Got unknown type {message}")


def get_cohere_chat_request(
    messages: List[BaseMessage],
    *,
    connectors: Optional[List[Dict[str, str]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get the request for the Cohere chat API.

    Args:
        messages: The messages.
        connectors: The connectors.
        **kwargs: The keyword arguments.

    Returns:
        The request for the Cohere chat API.
    """
    documents = (
        None
        if "source_documents" not in kwargs
        else [
            {
                "snippet": doc.page_content,
                "id": doc.metadata.get("id") or f"doc-{str(i)}",
            }
            for i, doc in enumerate(kwargs["source_documents"])
        ]
    )
    kwargs.pop("source_documents", None)
    maybe_connectors = connectors if documents is None else None

    # by enabling automatic prompt truncation, the probability of request failure is
    # reduced with minimal impact on response quality
    prompt_truncation = (
        "AUTO" if documents is not None or connectors is not None else None
    )

    return {
        "message": messages[-1].content,
        "chat_history": [
            {"role": get_role(x), "message": x.content} for x in messages[:-1]
        ],
        "documents": documents,
        "connectors": maybe_connectors,
        "prompt_truncation": prompt_truncation,
        **kwargs,
    }


class ChatCohere(BaseChatModel, BaseCohere):
    """`Cohere` chat large language models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatCohere
            from langchain.schema import HumanMessage

            chat = ChatCohere(model="foo")
            result = chat([HumanMessage(content="Hello")])
            print(result.content)
    """

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "cohere-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {
            "temperature": self.temperature,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        stream = self.client.chat(**request, stream=True)

        for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
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
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        stream = await self.async_client.chat(**request, stream=True)

        async for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    await run_manager.on_llm_new_token(delta)

    def _get_generation_info(self, response: Any) -> Dict[str, Any]:
        """Get the generation info from cohere API response."""
        return {
            "documents": response.documents,
            "citations": response.citations,
            "search_results": response.search_results,
            "search_queries": response.search_queries,
            "token_count": response.token_count,
        }

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
            return _generate_from_stream(stream_iter)

        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        response = self.client.chat(**request)

        message = AIMessage(content=response.text)
        generation_info = None
        if hasattr(response, "documents"):
            generation_info = self._get_generation_info(response)
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

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
            return await _agenerate_from_stream(stream_iter)

        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        response = self.client.chat(**request, stream=False)

        message = AIMessage(content=response.text)
        generation_info = None
        if hasattr(response, "documents"):
            generation_info = self._get_generation_info(response)
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        return len(self.client.tokenize(text).tokens)
