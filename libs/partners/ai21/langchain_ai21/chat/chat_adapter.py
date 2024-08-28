from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional, Union, cast, overload

from ai21.models import ChatMessage as J2ChatMessage
from ai21.models import RoleType
from ai21.models.chat import (
    AssistantMessage as AI21AssistantMessage,
)
from ai21.models.chat import ChatCompletionChunk, ChatMessageParam
from ai21.models.chat import ChatMessage as AI21ChatMessage
from ai21.models.chat import SystemMessage as AI21SystemMessage
from ai21.models.chat import ToolCall as AI21ToolCall
from ai21.models.chat import ToolFunction as AI21ToolFunction
from ai21.models.chat import ToolMessage as AI21ToolMessage
from ai21.models.chat import UserMessage as AI21UserMessage
from ai21.stream.stream import Stream as AI21Stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers.openai_tools import parse_tool_call
from langchain_core.outputs import ChatGenerationChunk

_ChatMessageTypes = Union[AI21ChatMessage, J2ChatMessage]
_SYSTEM_ERR_MESSAGE = "System message must be at beginning of message list."
_ROLE_TYPE = Union[str, RoleType]


class ChatAdapter(ABC):
    """Common interface for the different Chat models available in AI21.

    It converts LangChain messages to AI21 messages.
    Calls the appropriate AI21 model API with the converted messages.
    """

    @abstractmethod
    def convert_messages(
        self,
        messages: List[BaseMessage],
    ) -> Dict[str, Any]:
        pass

    def _convert_message_to_ai21_message(
        self,
        message: BaseMessage,
    ) -> _ChatMessageTypes:
        role = self._parse_role(message)
        return self._chat_message(role=role, message=message)

    def _parse_role(self, message: BaseMessage) -> _ROLE_TYPE:
        role = None

        if isinstance(message, SystemMessage):
            return RoleType.SYSTEM

        if isinstance(message, HumanMessage):
            return RoleType.USER

        if isinstance(message, AIMessage):
            return RoleType.ASSISTANT

        if isinstance(message, ToolMessage):
            return RoleType.TOOL

        if isinstance(self, J2ChatAdapter):
            if not role:
                raise ValueError(
                    f"Could not resolve role type from message {message}. "
                    f"Only support {HumanMessage.__name__} and {AIMessage.__name__}."
                )

        # if it gets here, we rely on the server to handle the role type
        return message.type

    @abstractmethod
    def _chat_message(
        self,
        role: _ROLE_TYPE,
        message: BaseMessage,
    ) -> _ChatMessageTypes:
        pass

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[True],
        **params: Any,
    ) -> Iterator[ChatGenerationChunk]:
        pass

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[False],
        **params: Any,
    ) -> List[BaseMessage]:
        pass

    @abstractmethod
    def call(
        self,
        client: Any,
        stream: Literal[True] | Literal[False],
        **params: Any,
    ) -> List[BaseMessage] | Iterator[ChatGenerationChunk]:
        pass

    def _get_system_message_from_message(self, message: BaseMessage) -> str:
        if not isinstance(message.content, str):
            raise ValueError(
                f"System Message must be of type str. Got {type(message.content)}"
            )

        return message.content


class J2ChatAdapter(ChatAdapter):
    """Adapter for J2Chat models."""

    def convert_messages(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        system_message = ""
        converted_messages = []  # type: ignore

        for i, message in enumerate(messages):
            if message.type == "system":
                if i != 0:
                    raise ValueError(_SYSTEM_ERR_MESSAGE)
                else:
                    system_message = self._get_system_message_from_message(message)
            else:
                converted_message = self._convert_message_to_ai21_message(message)
                converted_messages.append(converted_message)

        return {"system": system_message, "messages": converted_messages}

    def _chat_message(
        self,
        role: _ROLE_TYPE,
        message: BaseMessage,
    ) -> J2ChatMessage:
        return J2ChatMessage(role=RoleType(role), text=cast(str, message.content))

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[True],
        **params: Any,
    ) -> Iterator[ChatGenerationChunk]: ...

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[False],
        **params: Any,
    ) -> List[BaseMessage]: ...

    def call(
        self,
        client: Any,
        stream: Literal[True] | Literal[False],
        **params: Any,
    ) -> List[BaseMessage] | Iterator[ChatGenerationChunk]:
        if stream:
            raise NotImplementedError("Streaming is not supported for Jurassic models.")

        response = client.chat.create(**params)

        return [AIMessage(output.text) for output in response.outputs]


class JambaChatCompletionsAdapter(ChatAdapter):
    """Adapter for Jamba Chat Completions."""

    def convert_messages(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        return {
            "messages": [
                self._convert_message_to_ai21_message(message) for message in messages
            ],
        }

    def _convert_lc_tool_calls_to_ai21_tool_calls(
        self, tool_calls: List[ToolCall]
    ) -> Optional[List[AI21ToolCall]]:
        """
        Convert Langchain ToolCalls to AI21 ToolCalls.
        """
        ai21_tool_calls: List[AI21ToolCall] = []
        for lc_tool_call in tool_calls:
            if "id" not in lc_tool_call or not lc_tool_call["id"]:
                raise ValueError("Tool call ID is missing or empty.")

            ai21_tool_call = AI21ToolCall(
                id=lc_tool_call["id"],
                type="function",
                function=AI21ToolFunction(
                    name=lc_tool_call["name"],
                    arguments=str(lc_tool_call["args"]),
                ),
            )
            ai21_tool_calls.append(ai21_tool_call)

        return ai21_tool_calls

    def _get_content_as_string(self, base_message: BaseMessage) -> str:
        if isinstance(base_message.content, str):
            return base_message.content
        elif isinstance(base_message.content, list):
            return "\n".join(str(item) for item in base_message.content)
        else:
            raise ValueError("Unsupported content type")

    def _chat_message(
        self,
        role: _ROLE_TYPE,
        message: BaseMessage,
    ) -> ChatMessageParam:
        content = self._get_content_as_string(message)

        if isinstance(message, AIMessage):
            return AI21AssistantMessage(
                tool_calls=self._convert_lc_tool_calls_to_ai21_tool_calls(
                    message.tool_calls
                ),
                content=content or None,
            )
        if isinstance(message, ToolMessage):
            return AI21ToolMessage(
                tool_call_id=message.tool_call_id,
                content=content,
            )
        if isinstance(message, HumanMessage):
            return AI21UserMessage(
                content=content,
            )
        if isinstance(message, SystemMessage):
            return AI21SystemMessage(
                content=content,
            )
        return AI21ChatMessage(
            role=role.value if isinstance(role, RoleType) else role,
            content=content,
        )

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[True],
        **params: Any,
    ) -> Iterator[ChatGenerationChunk]: ...

    @overload
    def call(
        self,
        client: Any,
        stream: Literal[False],
        **params: Any,
    ) -> List[BaseMessage]: ...

    def call(
        self,
        client: Any,
        stream: Literal[True] | Literal[False],
        **params: Any,
    ) -> List[BaseMessage] | Iterator[ChatGenerationChunk]:
        response = client.chat.completions.create(stream=stream, **params)

        if stream:
            return self._stream_response(response)

        ai_messages: List[BaseMessage] = []
        for message in response.choices:
            if message.message.tool_calls:
                tool_calls = [
                    parse_tool_call(tool_call.model_dump(), return_id=True)
                    for tool_call in message.message.tool_calls
                ]
                ai_messages.append(AIMessage("", tool_calls=tool_calls))
            else:
                ai_messages.append(AIMessage(message.message.content))

        return ai_messages

    def _stream_response(
        self,
        response: AI21Stream[ChatCompletionChunk],
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in response:
            converted_message = self._convert_ai21_chunk_to_chunk(chunk)
            yield ChatGenerationChunk(message=converted_message)

    def _convert_ai21_chunk_to_chunk(
        self,
        chunk: ChatCompletionChunk,
    ) -> BaseMessageChunk:
        usage = chunk.usage
        content = chunk.choices[0].delta.content or ""

        if usage is None:
            return AIMessageChunk(
                content=content,
            )

        return AIMessageChunk(
            content=content,
            usage_metadata=UsageMetadata(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )
