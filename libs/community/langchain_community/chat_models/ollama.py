import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_community.llms.ollama import OllamaEndpointNotFoundError, _OllamaCommon


@deprecated("0.0.3", alternative="_chat_stream_response_to_chat_generation_chunk")
def _stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(content=parsed_response.get("response", "")),
        generation_info=generation_info,
    )


def _chat_stream_response_to_chat_generation_chunk(
    stream_response: str,
) -> ChatGenerationChunk:
    """Convert a stream response to a generation chunk."""
    parsed_response = json.loads(stream_response)
    generation_info = parsed_response if parsed_response.get("done") is True else None
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=parsed_response.get("message", {}).get("content", "")
        ),
        generation_info=generation_info,
    )


class ChatOllama(BaseChatModel, _OllamaCommon):
    """Ollama locally runs large language models.

    To use, follow the instructions at https://ollama.ai/.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatOllama
            ollama = ChatOllama(model="llama2")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "ollama-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @deprecated("0.0.3", alternative="_convert_messages_to_ollama_messages")
    def _format_message_as_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            if isinstance(message.content, List):
                first_content = cast(List[Dict], message.content)[0]
                content_type = first_content.get("type")
                if content_type == "text":
                    message_text = f"[INST] {first_content['text']} [/INST]"
                elif content_type == "image_url":
                    message_text = first_content["image_url"]["url"]
            else:
                message_text = f"[INST] {message.content} [/INST]"
        elif isinstance(message, AIMessage):
            message_text = f"{message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"<<SYS>> {message.content} <</SYS>>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            [self._format_message_as_text(message) for message in messages]
        )

    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Union[str, List[str]]]]:
        ollama_messages: List = []
        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "image_url":
                        if isinstance(content_part.get("image_url"), str):
                            image_url_components = content_part["image_url"].split(",")
                            # Support data:image/jpeg;base64,<image> format
                            # and base64 strings
                            if len(image_url_components) > 1:
                                images.append(image_url_components[1])
                            else:
                                images.append(image_url_components[0])
                        else:
                            raise ValueError(
                                "Only string image_url " "content parts are supported."
                            )
                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )

            ollama_messages.append(
                {
                    "role": role,
                    "content": content,
                    "images": images,
                }
            )

        return ollama_messages

    def _create_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        payload = {
            "messages": self._convert_messages_to_ollama_messages(messages),
        }
        yield from self._create_stream(
            payload=payload, stop=stop, api_url=f"{self.base_url}/api/chat/", **kwargs
        )

    async def _acreate_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload = {
            "messages": self._convert_messages_to_ollama_messages(messages),
        }
        async for stream_resp in self._acreate_stream(
            payload=payload, stop=stop, api_url=f"{self.base_url}/api/chat/", **kwargs
        ):
            yield stream_resp

    def _chat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[ChatGenerationChunk] = None
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk: Optional[ChatGenerationChunk] = None
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Ollama's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = ollama([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """

        final_chunk = self._chat_stream_with_aggregation(
            messages,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **kwargs,
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Ollama's generate endpoint.

        Args:
            messages: The list of base messages to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            Chat generations from the model

        Example:
            .. code-block:: python

                response = ollama([
                    HumanMessage(content="Tell me about the history of AI")
                ])
        """

        final_chunk = await self._achat_stream_with_aggregation(
            messages,
            stop=stop,
            run_manager=run_manager,
            verbose=self.verbose,
            **kwargs,
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
                if stream_resp:
                    chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            chunk.text,
                            verbose=self.verbose,
                        )
                    yield chunk
        except OllamaEndpointNotFoundError:
            yield from self._legacy_stream(messages, stop, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if stream_resp:
                chunk = _chat_stream_response_to_chat_generation_chunk(stream_resp)
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    @deprecated("0.0.3", alternative="_stream")
    def _legacy_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._format_messages_as_text(messages)
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if stream_resp:
                chunk = _stream_response_to_chat_generation_chunk(stream_resp)
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk
