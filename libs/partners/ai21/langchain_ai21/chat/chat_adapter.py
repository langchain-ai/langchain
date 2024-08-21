from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Union, cast, overload, Optional

from ai21 import AI21Client
from ai21.models import RoleType, ChatMessage as J2ChatMessage
from ai21.models.chat import (ChatCompletionChunk, ToolCall as AI21ToolCall, ToolFunction as AI21ToolFunction,
                              ToolMessage as AI21ToolMessage, AssistantMessage as AI21AssistantMessage,
                              UserMessage as AI21UserMessage, SystemMessage as AI21SystemMessage,
                              ChatMessage as AI21ChatMessage)

from ai21.models.chat.chat_message import ChatMessageParam
from ai21.stream.stream import Stream as AI21Stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage, SystemMessage, ToolMessage
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

    def convert_lc_tool_calls_to_ai21_tool_call(
            self, tool_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[AI21ToolCall]]:
        """ Currently, AI21 supports only function type tool calls. We need to Convert all the
        Langchain ToolCall: {'args': Dict[str, Any], 'id': str, 'name': str, 'type': 'tool_call'} to
        AI21 ToolCall: {'function': {'name': str, arguments: str}, 'id': str, 'type': 'function'} """
        return [AI21ToolCall(
            id=tool_call["id"],
            type="function",
            function=AI21ToolFunction(name=tool_call["name"], arguments=str(tool_call["args"])),
        ) for tool_call in tool_calls] if tool_calls else None

    def _chat_message(
        self,
        role: _ROLE_TYPE,
        message: BaseMessage,
    ) -> ChatMessageParam:
        if role == RoleType.ASSISTANT:
            return AI21AssistantMessage(
                tool_calls=self.convert_lc_tool_calls_to_ai21_tool_call(message.tool_calls),
                content=None if message.content == "" else message.content,
            )
        if role == RoleType.TOOL:
            return AI21ToolMessage(
                tool_call_id=message.tool_call_id,
                content=message.content,
            )
        if role == RoleType.USER:
            return AI21UserMessage(
                content=message.content,
            )
        if role == RoleType.SYSTEM:
            return AI21SystemMessage(
                content=message.content,
            )
        return AI21ChatMessage(
            role=role.value if isinstance(role, RoleType) else role,
            content=message.content,
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
        client: AI21Client,
        stream: Literal[True] | Literal[False],
        **params: Any,
    ) -> List[BaseMessage] | Iterator[ChatGenerationChunk]:
        response = client.chat.completions.create(stream=stream, **params)
        if stream:
            return self._stream_response(response)

        ai_messages = []
        for message in response.choices:
            if not message.message.tool_calls:
                ai_messages.append(AIMessage(message.message.content))
            else:
                tool_calls = [parse_tool_call(tool_call.model_dump(), return_id=True)
                              for tool_call in message.message.tool_calls]
                ai_messages.append(AIMessage("", tool_calls=tool_calls))

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
