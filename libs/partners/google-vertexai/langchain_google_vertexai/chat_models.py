"""Wrapper around Google VertexAI chat-based models."""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union, cast
from urllib.parse import urlparse

import requests
from google.cloud.aiplatform_v1beta1.types.content import Part as GapicPart
from google.cloud.aiplatform_v1beta1.types.tool import FunctionCall
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from vertexai.language_models import (  # type: ignore
    ChatMessage,
    ChatModel,
    ChatSession,
    CodeChatModel,
    CodeChatSession,
    InputOutputTextPair,
)
from vertexai.preview.generative_models import (  # type: ignore
    Candidate,
    Content,
    GenerativeModel,
    Image,
    Part,
)

from langchain_google_vertexai._utils import (
    get_generation_info,
    is_codey_model,
    is_gemini_model,
    load_image_from_gcs,
)
from langchain_google_vertexai.functions_utils import (
    _format_tools_to_vertex_tool,
)
from langchain_google_vertexai.llms import (
    _VertexAICommon,
)

logger = logging.getLogger(__name__)


@dataclass
class _ChatHistory:
    """Represents a context and a history of messages."""

    history: List[ChatMessage] = field(default_factory=list)
    context: Optional[str] = None


def _parse_chat_history(history: List[BaseMessage]) -> _ChatHistory:
    """Parse a sequence of messages into history.

    Args:
        history: The list of messages to re-create the history of the chat.
    Returns:
        A parsed chat history.
    Raises:
        ValueError: If a sequence of message has a SystemMessage not at the
        first place.
    """

    vertex_messages, context = [], None
    for i, message in enumerate(history):
        content = cast(str, message.content)
        if i == 0 and isinstance(message, SystemMessage):
            context = content
        elif isinstance(message, AIMessage):
            vertex_message = ChatMessage(content=message.content, author="bot")
            vertex_messages.append(vertex_message)
        elif isinstance(message, HumanMessage):
            vertex_message = ChatMessage(content=message.content, author="user")
            vertex_messages.append(vertex_message)
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )
    chat_history = _ChatHistory(context=context, history=vertex_messages)
    return chat_history


def _is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _parse_chat_history_gemini(
    history: List[BaseMessage],
    project: Optional[str] = None,
    convert_system_message_to_human: Optional[bool] = False,
) -> List[Content]:
    def _convert_to_prompt(part: Union[str, Dict]) -> Part:
        if isinstance(part, str):
            return Part.from_text(part)

        if not isinstance(part, Dict):
            raise ValueError(
                f"Message's content is expected to be a dict, got {type(part)}!"
            )
        if part["type"] == "text":
            return Part.from_text(part["text"])
        elif part["type"] == "image_url":
            path = part["image_url"]["url"]
            if path.startswith("gs://"):
                image = load_image_from_gcs(path=path, project=project)
            elif path.startswith("data:image/"):
                # extract base64 component from image uri
                try:
                    regexp = r"data:image/\w{2,4};base64,(.*)"
                    encoded = re.search(regexp, path).group(1)  # type: ignore
                except AttributeError:
                    raise ValueError(
                        "Invalid image uri. It should be in the format "
                        "data:image/<image_type>;base64,<base64_encoded_image>."
                    )
                image = Image.from_bytes(base64.b64decode(encoded))
            elif _is_url(path):
                response = requests.get(path)
                response.raise_for_status()
                image = Image.from_bytes(response.content)
            else:
                image = Image.load_from_file(path)
        else:
            raise ValueError("Only text and image_url types are supported!")
        return Part.from_image(image)

    def _convert_to_parts(message: BaseMessage) -> List[Part]:
        raw_content = message.content
        if isinstance(raw_content, str):
            raw_content = [raw_content]
        return [_convert_to_prompt(part) for part in raw_content]

    vertex_messages = []
    raw_system_message = None
    for i, message in enumerate(history):
        if (
            i == 0
            and isinstance(message, SystemMessage)
            and not convert_system_message_to_human
        ):
            raise ValueError(
                """SystemMessages are not yet supported!
                
To automatically convert the leading SystemMessage to a HumanMessage,
set  `convert_system_message_to_human` to True. Example:

llm = ChatVertexAI(model_name="gemini-pro", convert_system_message_to_human=True)
"""
            )
        elif i == 0 and isinstance(message, SystemMessage):
            raw_system_message = message
            continue
        elif isinstance(message, AIMessage):
            raw_function_call = message.additional_kwargs.get("function_call")
            role = "model"
            if raw_function_call:
                function_call = FunctionCall(
                    {
                        "name": raw_function_call["name"],
                        "args": json.loads(raw_function_call["arguments"]),
                    }
                )
                gapic_part = GapicPart(function_call=function_call)
                parts = [Part._from_gapic(gapic_part)]
            else:
                parts = _convert_to_parts(message)
        elif isinstance(message, HumanMessage):
            role = "user"
            parts = _convert_to_parts(message)
        elif isinstance(message, FunctionMessage):
            role = "user"
            parts = [
                Part.from_function_response(
                    name=message.name,
                    response={
                        "content": message.content,
                    },
                )
            ]
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )

        if raw_system_message:
            if role == "model":
                raise ValueError(
                    "SystemMessage should be followed by a HumanMessage and "
                    "not by AIMessage."
                )
            parts = _convert_to_parts(raw_system_message) + parts
            raw_system_message = None

        vertex_message = Content(role=role, parts=parts)
        vertex_messages.append(vertex_message)
    return vertex_messages


def _parse_examples(examples: List[BaseMessage]) -> List[InputOutputTextPair]:
    if len(examples) % 2 != 0:
        raise ValueError(
            f"Expect examples to have an even amount of messages, got {len(examples)}."
        )
    example_pairs = []
    input_text = None
    for i, example in enumerate(examples):
        if i % 2 == 0:
            if not isinstance(example, HumanMessage):
                raise ValueError(
                    f"Expected the first message in a part to be from human, got "
                    f"{type(example)} for the {i}th message."
                )
            input_text = example.content
        if i % 2 == 1:
            if not isinstance(example, AIMessage):
                raise ValueError(
                    f"Expected the second message in a part to be from AI, got "
                    f"{type(example)} for the {i}th message."
                )
            pair = InputOutputTextPair(
                input_text=input_text, output_text=example.content
            )
            example_pairs.append(pair)
    return example_pairs


def _get_question(messages: List[BaseMessage]) -> HumanMessage:
    """Get the human message at the end of a list of input messages to a chat model."""
    if not messages:
        raise ValueError("You should provide at least one message to start the chat!")
    question = messages[-1]
    if not isinstance(question, HumanMessage):
        raise ValueError(
            f"Last message in the list should be from human, got {question.type}."
        )
    return question


def _parse_response_candidate(response_candidate: "Candidate") -> AIMessage:
    try:
        content = response_candidate.text
    except ValueError:
        content = ""

    additional_kwargs = {}
    first_part = response_candidate.content.parts[0]
    if first_part.function_call:
        function_call = {"name": first_part.function_call.name}

        # dump to match other function calling llm for now
        function_call["arguments"] = json.dumps(
            {k: first_part.function_call.args[k] for k in first_part.function_call.args}
        )
        additional_kwargs["function_call"] = function_call
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


class ChatVertexAI(_VertexAICommon, BaseChatModel):
    """`Vertex AI` Chat large language models API."""

    model_name: str = "chat-bison"
    "Underlying model name."
    examples: Optional[List[BaseMessage]] = None
    convert_system_message_to_human: bool = False
    """Whether to merge any leading SystemMessage into the following HumanMessage.
    
    Gemini does not support system messages; any unsupported messages will 
    raise an error."""

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "vertexai"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        is_gemini = is_gemini_model(values["model_name"])
        safety_settings = values["safety_settings"]

        if safety_settings and not is_gemini:
            raise ValueError("Safety settings are only supported for Gemini models")

        cls._init_vertexai(values)
        if is_gemini:
            values["client"] = GenerativeModel(
                model_name=values["model_name"], safety_settings=safety_settings
            )
        else:
            if is_codey_model(values["model_name"]):
                model_cls = CodeChatModel
            else:
                model_cls = ChatModel
            values["client"] = model_cls.from_pretrained(values["model_name"])
        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: The list of stop words (optional).
            run_manager: The CallbackManager for LLM run, it's not used at the moment.
            stream: Whether to use the streaming endpoint.

        Returns:
            The ChatResult that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        should_stream = stream if stream is not None else self.streaming
        safety_settings = kwargs.pop("safety_settings", None)
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        params = self._prepare_params(stop=stop, stream=False, **kwargs)
        msg_params = {}
        if "candidate_count" in params:
            msg_params["candidate_count"] = params.pop("candidate_count")

        if self._is_gemini_model:
            history_gemini = _parse_chat_history_gemini(
                messages,
                project=self.project,
                convert_system_message_to_human=self.convert_system_message_to_human,
            )
            message = history_gemini.pop()
            chat = self.client.start_chat(history=history_gemini)

            # set param to `functions` until core tool/function calling implemented
            raw_tools = params.pop("functions") if "functions" in params else None
            tools = _format_tools_to_vertex_tool(raw_tools) if raw_tools else None
            response = chat.send_message(
                message,
                generation_config=params,
                tools=tools,
                safety_settings=safety_settings,
            )
            generations = [
                ChatGeneration(
                    message=_parse_response_candidate(c),
                    generation_info=get_generation_info(c, self._is_gemini_model),
                )
                for c in response.candidates
            ]
        else:
            question = _get_question(messages)
            history = _parse_chat_history(messages[:-1])
            examples = kwargs.get("examples") or self.examples
            if examples:
                params["examples"] = _parse_examples(examples)
            chat = self._start_chat(history, **params)
            response = chat.send_message(question.content, **msg_params)
            generations = [
                ChatGeneration(
                    message=AIMessage(content=r.text),
                    generation_info=get_generation_info(r, self._is_gemini_model),
                )
                for r in response.candidates
            ]
        return ChatResult(generations=generations)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: The list of stop words (optional).
            run_manager: The CallbackManager for LLM run, it's not used at the moment.

        Returns:
            The ChatResult that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        if "stream" in kwargs:
            kwargs.pop("stream")
            logger.warning("ChatVertexAI does not currently support async streaming.")

        params = self._prepare_params(stop=stop, **kwargs)
        safety_settings = kwargs.pop("safety_settings", None)
        msg_params = {}
        if "candidate_count" in params:
            msg_params["candidate_count"] = params.pop("candidate_count")

        if self._is_gemini_model:
            history_gemini = _parse_chat_history_gemini(
                messages,
                project=self.project,
                convert_system_message_to_human=self.convert_system_message_to_human,
            )
            message = history_gemini.pop()
            chat = self.client.start_chat(history=history_gemini)
            # set param to `functions` until core tool/function calling implemented
            raw_tools = params.pop("functions") if "functions" in params else None
            tools = _format_tools_to_vertex_tool(raw_tools) if raw_tools else None
            response = await chat.send_message_async(
                message,
                generation_config=params,
                tools=tools,
                safety_settings=safety_settings,
            )
            generations = [
                ChatGeneration(
                    message=_parse_response_candidate(c),
                    generation_info=get_generation_info(c, self._is_gemini_model),
                )
                for c in response.candidates
            ]
        else:
            question = _get_question(messages)
            history = _parse_chat_history(messages[:-1])
            examples = kwargs.get("examples", None) or self.examples
            if examples:
                params["examples"] = _parse_examples(examples)
            chat = self._start_chat(history, **params)
            response = await chat.send_message_async(question.content, **msg_params)
            generations = [
                ChatGeneration(
                    message=AIMessage(content=r.text),
                    generation_info=get_generation_info(r, self._is_gemini_model),
                )
                for r in response.candidates
            ]
        return ChatResult(generations=generations)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        if self._is_gemini_model:
            history_gemini = _parse_chat_history_gemini(
                messages,
                project=self.project,
                convert_system_message_to_human=self.convert_system_message_to_human,
            )
            message = history_gemini.pop()
            chat = self.client.start_chat(history=history_gemini)
            # set param to `functions` until core tool/function calling implemented
            raw_tools = params.pop("functions") if "functions" in params else None
            tools = _format_tools_to_vertex_tool(raw_tools) if raw_tools else None
            responses = chat.send_message(
                message, stream=True, generation_config=params, tools=tools
            )
            for response in responses:
                message = _parse_response_candidate(response.candidates[0])
                if run_manager:
                    run_manager.on_llm_new_token(message.content)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=message.content,
                        additional_kwargs=message.additional_kwargs,
                    )
                )
        else:
            question = _get_question(messages)
            history = _parse_chat_history(messages[:-1])
            examples = kwargs.get("examples", None)
            if examples:
                params["examples"] = _parse_examples(examples)
            chat = self._start_chat(history, **params)
            responses = chat.send_message_streaming(question.content, **params)
        for response in responses:
            if run_manager:
                run_manager.on_llm_new_token(response.text)
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=response.text),
                generation_info=get_generation_info(response, self._is_gemini_model),
            )

    def _start_chat(
        self, history: _ChatHistory, **kwargs: Any
    ) -> Union[ChatSession, CodeChatSession]:
        if not self.is_codey_model:
            return self.client.start_chat(
                context=history.context, message_history=history.history, **kwargs
            )
        else:
            return self.client.start_chat(message_history=history.history, **kwargs)
