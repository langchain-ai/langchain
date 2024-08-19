from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core._api.deprecation import deprecated
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

from langchain_community.llms.cohere import BaseCohere


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

    req = {
        "message": messages[-1].content,
        "chat_history": [
            {"role": get_role(x), "message": x.content} for x in messages[:-1]
        ],
        "documents": documents,
        "connectors": maybe_connectors,
        "prompt_truncation": prompt_truncation,
        **kwargs,
    }

    return {k: v for k, v in req.items() if v is not None}


@deprecated(
    since="0.0.30", removal="1.0", alternative_import="langchain_cohere.ChatCohere"
)
class ChatCohere(BaseChatModel, BaseCohere):
    """`Cohere` chat large language models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatCohere
            from langchain_core.messages import HumanMessage

            chat = ChatCohere(max_tokens=256, temperature=0.75)

            messages = [HumanMessage(content="knock knock")]
            chat.invoke(messages)
    """

    class Config:
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

        if hasattr(self.client, "chat_stream"):  # detect and support sdk v5
            stream = self.client.chat_stream(**request)
        else:
            stream = self.client.chat(**request, stream=True)

        for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)

        if hasattr(self.async_client, "chat_stream"):  # detect and support sdk v5
            stream = await self.async_client.chat_stream(**request)
        else:
            stream = await self.async_client.chat(**request, stream=True)

        async for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    await run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk

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
            return generate_from_stream(stream_iter)

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
            return await agenerate_from_stream(stream_iter)

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

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        return len(self.client.tokenize(text=text).tokens)
