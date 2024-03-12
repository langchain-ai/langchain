import json
import os
import posixpath
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Type

from httpx import (AsyncClient, AsyncHTTPTransport, Client, HTTPTransport,
                   Limits, Response)
from langchain_core.callbacks import (AsyncCallbackManagerForLLMRun,
                                      CallbackManagerForLLMRun)
from langchain_core.language_models.chat_models import (BaseChatModel,
                                                        agenerate_from_stream,
                                                        generate_from_stream)
from langchain_core.messages import (AIMessage, AIMessageChunk, BaseMessage,
                                     BaseMessageChunk, ChatMessage,
                                     ChatMessageChunk, HumanMessage,
                                     HumanMessageChunk, SystemMessageChunk)
from langchain_core.outputs import (ChatGeneration, ChatGenerationChunk,
                                    ChatResult)
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str


class ChatUnify(BaseChatModel):
    """ChatUnify chat model.

    Example:
        .. code-block:: python

            from langchain_unify import ChatUnify


            model = ChatUnify()
    """

    client: Client = Field(default=None)
    async_client: AsyncClient = Field(default=None)
    unify_api_key: Optional[SecretStr] = None
    unify_api_url: str = "https://api.unify.ai/v0/"
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 128

    model: str = "llama-2-70b-chat@lowest-input-cost"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    random_seed: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "unify-chat"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        unify_api_key = convert_to_secret_str(
            values.get("unify_api_key") or os.environ.get("UNIFY_API_KEY") or ""
        )
        values["unify_api_key"] = unify_api_key
        values["client"] = Client(
            follow_redirects=True,
            timeout=values.get("timeout"),
            transport=HTTPTransport(retries=values.get("max_retries")),
        )
        values["async_client"] = AsyncClient(
            follow_redirects=True,
            timeout=values.get("timeout"),
            limits=Limits(max_connections=values.get("max_concurrent_requests")),
            transport=AsyncHTTPTransport(retries=values.get("max_retries")),
        )
        return values

    def _check_response(self, response: Response) -> None:
        if response.status_code >= 500:
            raise Exception(f"Unify Server: Error {response.status}")
        elif response.status_code >= 400:
            raise ValueError(f"Unify received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"Unify returned an unexpected response with status "
                f"{response.status}: {response.text}"
            )

    def _get_request_headers(self, stream) -> Dict[str, str]:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Authorization": f"Bearer {self.unify_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in messages:
            formatted_message = {}
            if isinstance(message, ChatMessage):
                formatted_message["role"] = message.role
            elif isinstance(message, HumanMessage):
                formatted_message["role"] = "user"
            elif isinstance(message, AIMessage):
                formatted_message["role"] = "assistant"
            formatted_message["content"] = message.content
            formatted_messages.append(formatted_message)
        return formatted_messages

    def _convert_delta_to_message_chunk(
        self, _delta: Dict, default_class: Type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        role = _delta.get("role")
        content = _delta.get("content", "")
        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            return AIMessageChunk(content=content)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)
        else:
            return default_class(content=content)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        headers = self._get_request_headers(True)
        if "model" not in kwargs:
            kwargs["model"] = self.model
        body = {"messages": self._format_messages(messages), "stream": True, **kwargs}
        url = posixpath.join(self.unify_api_url, "chat/completions")
        with self.client.stream("post", url, headers=headers, json=body) as response:
            self._check_response(response)
            default_chunk_class = AIMessageChunk
            for line in response.iter_lines():
                if not line:
                    continue
                response_dict = line.removeprefix("data: ")
                response_json = json.loads(response_dict)
                choices = response_json["choices"][0]
                if not choices:
                    continue
                delta = choices["delta"]
                chunk = self._convert_delta_to_message_chunk(delta, default_chunk_class)
                default_chunk_class = chunk.__class__

                if run_manager:
                    run_manager.on_llm_new_token(token=chunk.content, chunk=chunk)
                yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        headers = self._get_request_headers(True)
        body = {"messages": self._formatted_messages(messages), **kwargs}
        url = posixpath.join(self.unify_api_url, "chat/completions")
        async with self.client.stream(
            "post", url, headers=headers, json=body
        ) as response:
            self._check_response(response)
            async for line in response.aiter_lines():
                if not line:
                    continue
                response_dict = line.removeprefix("data: ")
                response_json = json.loads(response_dict)
                delta = response_json["choices"][0]["delta"]
                content = delta["content"]
                if not content:
                    continue
                chunk = ChatGenerationChunk(message=AIMessage(content=content))
                if run_manager:
                    await run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=data["choices"][0]["message"]["content"])
                )
            ],
            llm_output=data,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        headers = self._get_request_headers(False)
        if "model" not in kwargs:
            kwargs["model"] = self.model
        body = {"messages": self._format_messages(messages), **kwargs}
        url = posixpath.join(self.unify_api_url, "chat/completions")
        response = self.client.post(url, headers=headers, json=body)
        self._check_response(response)
        return self._format_output(response.json(), **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        headers = self._get_request_headers(False)
        if "model" not in kwargs:
            kwargs["model"] = self.model
        body = {"messages": self._format_messages(messages), **kwargs}
        url = posixpath.join(self.unify_api_url, "chat/completions")
        response = await self.async_client.post(url, headers=headers, json=body)
        self._check_response(response)
        return self._format_output(response.json(), **kwargs)
