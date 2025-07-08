"""Hugging Face Chat Wrapper."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Callable, Literal, Optional, Union, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline


@dataclass
class TGI_RESPONSE:
    """Response from the TextGenInference API."""

    choices: list[Any]
    usage: dict


@dataclass
class TGI_MESSAGE:
    """Message to send to the TextGenInference API."""

    role: str
    content: str
    tool_calls: list[dict]


def _lc_tool_call_to_hf_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_hf_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.

    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_hf_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_hf_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        # If tool calls only, content is None not empty string
        if "tool_calls" in message_dict and message_dict["content"] == "":
            message_dict["content"] = None
        else:
            pass
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.

    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=_dict.get("name", "")
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
            additional_kwargs=additional_kwargs,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role or "")


def _is_huggingface_hub(llm: Any) -> bool:
    try:
        from langchain_community.llms.huggingface_hub import (
            HuggingFaceHub,  # type: ignore[import-not-found]
        )

        return isinstance(llm, HuggingFaceHub)
    except ImportError:
        # if no langchain community, it is not a HuggingFaceHub
        return False


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choice = chunk["choices"][0]
    _dict = choice["delta"]
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    tool_call_chunks: list[ToolCallChunk] = []
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for rtc in raw_tool_calls:
            with contextlib.suppress(KeyError):
                tool_call_chunks.append(
                    create_tool_call_chunk(
                        name=rtc["function"].get("name"),
                        args=rtc["function"].get("arguments"),
                        id=rtc.get("id"),
                        index=rtc.get("index"),
                    )
                )
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        if usage := chunk.get("usage"):
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
            }
        else:
            usage_metadata = None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _is_huggingface_textgen_inference(llm: Any) -> bool:
    try:
        from langchain_community.llms.huggingface_text_gen_inference import (
            HuggingFaceTextGenInference,  # type: ignore[import-not-found]
        )

        return isinstance(llm, HuggingFaceTextGenInference)
    except ImportError:
        # if no langchain community, it is not a HuggingFaceTextGenInference
        return False


def _is_huggingface_endpoint(llm: Any) -> bool:
    return isinstance(llm, HuggingFaceEndpoint)


def _is_huggingface_pipeline(llm: Any) -> bool:
    return isinstance(llm, HuggingFacePipeline)


class ChatHuggingFace(BaseChatModel):
    r"""Hugging Face LLM's as ChatModels.

    Works with `HuggingFaceTextGenInference`, `HuggingFaceEndpoint`,
    `HuggingFaceHub`, and `HuggingFacePipeline` LLMs.

    Upon instantiating this class, the model_id is resolved from the url
    provided to the LLM, and the appropriate tokenizer is loaded from
    the HuggingFace Hub.

    Setup:
        Install ``langchain-huggingface`` and ensure your Hugging Face token
        is saved.

        .. code-block:: bash

            pip install langchain-huggingface

        .. code-block:: python

            from huggingface_hub import login
            login() # You will be prompted for your HF key, which will then be saved locally

    Key init args — completion params:
        llm: `HuggingFaceTextGenInference`, `HuggingFaceEndpoint`, `HuggingFaceHub`, or
            'HuggingFacePipeline' LLM to be used.

    Key init args — client params:
        custom_get_token_ids: Optional[Callable[[str], list[int]]]
            Optional encoder to use for counting tokens.
        metadata: Optional[dict[str, Any]]
            Metadata to add to the run trace.
        tags: Optional[list[str]]
            Tags to add to the run trace.
        tokenizer: Any
        verbose: bool
            Whether to print out response text.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEndpoint,
            ChatHuggingFace

            llm = HuggingFaceEndpoint(
                repo_id="microsoft/Phi-3-mini-4k-instruct",
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            )

            chat = ChatHuggingFace(llm=llm, verbose=True)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user
                sentence to French."),
                ("human", "I love programming."),
            ]

            chat(...).invoke(messages)

        .. code-block:: python

            AIMessage(content='Je ai une passion pour le programme.\n\nIn
            French, we use "ai" for masculine subjects and "a" for feminine
            subjects. Since "programming" is gender-neutral in English, we
            will go with the masculine "programme".\n\nConfirmation: "J\'aime
            le programme." is more commonly used. The sentence above is
            technically accurate, but less commonly used in spoken French as
            "ai" is used less frequently in everyday speech.',
            response_metadata={'token_usage': ChatCompletionOutputUsage
            (completion_tokens=100, prompt_tokens=55, total_tokens=155),
            'model': '', 'finish_reason': 'length'},
            id='run-874c24b7-0272-4c99-b259-5d6d7facbc56-0')

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='Je ai une passion pour le programme.\n\nIn French, we use
            "ai" for masculine subjects and "a" for feminine subjects.
            Since "programming" is gender-neutral in English,
            we will go with the masculine "programme".\n\nConfirmation:
            "J\'aime le programme." is more commonly used. The sentence
            above is technically accurate, but less commonly used in spoken
            French as "ai" is used less frequently in everyday speech.'
            response_metadata={'token_usage': ChatCompletionOutputUsage
            (completion_tokens=100, prompt_tokens=55, total_tokens=155),
            'model': '', 'finish_reason': 'length'}
            id='run-7d7b1967-9612-4f9a-911a-b2b5ca85046a-0'

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

        .. code-block:: python

            AIMessage(content='Je déaime le programming.\n\nLittérale : Je
            (j\'aime) déaime (le) programming.\n\nNote: "Programming" in
            French is "programmation". But here, I used "programming" instead
            of "programmation" because the user said "I love programming"
            instead of "I love programming (in French)", which would be
            "J\'aime la programmation". By translating the sentence
            literally, I preserved the original meaning of the user\'s
            sentence.', id='run-fd850318-e299-4735-b4c6-3496dc930b1d-0')

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state,
                e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state,
                e.g. San Francisco, CA")

            chat_with_tools = chat.bind_tools([GetWeather, GetPopulation])
            ai_msg = chat_with_tools.invoke("Which city is hotter today and
            which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': '0'}]

    Response metadata
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python
            {'token_usage': ChatCompletionOutputUsage(completion_tokens=100,
            prompt_tokens=8, total_tokens=108),
             'model': '',
             'finish_reason': 'length'}

    """  # noqa: E501

    llm: Any
    """LLM, must be of type HuggingFaceTextGenInference, HuggingFaceEndpoint,
        HuggingFaceHub, or HuggingFacePipeline."""
    tokenizer: Any = None
    """Tokenizer for the model. Only used for HuggingFacePipeline."""
    model_id: Optional[str] = None
    """Model ID for the model. Only used for HuggingFaceEndpoint."""
    temperature: Optional[float] = None
    """What sampling temperature to use."""
    stop: Optional[Union[str, list[str]]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    presence_penalty: Optional[float] = None
    """Penalizes repeated tokens."""
    frequency_penalty: Optional[float] = None
    """Penalizes repeated tokens according to frequency."""
    seed: Optional[int] = None
    """Seed for generation"""
    logprobs: Optional[bool] = None
    """Whether to return logprobs."""
    top_logprobs: Optional[int] = None
    """Number of most likely tokens to return at each token position, each with
     an associated log probability. `logprobs` must be set to true
     if this parameter is used."""
    logit_bias: Optional[dict[int, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: Optional[int] = None
    """Number of chat completions to generate for each prompt."""
    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._resolve_model_id()

    @model_validator(mode="after")
    def validate_llm(self) -> Self:
        if (
            not _is_huggingface_hub(self.llm)
            and not _is_huggingface_textgen_inference(self.llm)
            and not _is_huggingface_endpoint(self.llm)
            and not _is_huggingface_pipeline(self.llm)
        ):
            msg = (
                "Expected llm to be one of HuggingFaceTextGenInference, "
                "HuggingFaceEndpoint, HuggingFaceHub, HuggingFacePipeline "
                f"received {type(self.llm)}"
            )
            raise TypeError(msg)
        return self

    def _create_chat_result(self, response: dict) -> ChatResult:
        generations = []
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            generation_info = {"finish_reason": res.get("finish_reason")}
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_id,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming

        if _is_huggingface_textgen_inference(self.llm):
            message_dicts, params = self._create_message_dicts(messages, stop)
            answer = self.llm.client.chat(messages=message_dicts, **kwargs)
            return self._create_chat_result(answer)
        if _is_huggingface_endpoint(self.llm):
            if should_stream:
                stream_iter = self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return generate_from_stream(stream_iter)
            message_dicts, params = self._create_message_dicts(messages, stop)
            params = {
                "stop": stop,
                **params,
                **({"stream": stream} if stream is not None else {}),
                **kwargs,
            }
            answer = self.llm.client.chat_completion(messages=message_dicts, **params)
            return self._create_chat_result(answer)
        llm_input = self._to_chat_prompt(messages)

        if should_stream:
            stream_iter = self.llm._stream(
                llm_input, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        if _is_huggingface_textgen_inference(self.llm):
            message_dicts, params = self._create_message_dicts(messages, stop)
            answer = await self.llm.async_client.chat(messages=message_dicts, **kwargs)
            return self._create_chat_result(answer)
        if _is_huggingface_endpoint(self.llm):
            should_stream = stream if stream is not None else self.streaming
            if should_stream:
                stream_iter = self._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return await agenerate_from_stream(stream_iter)
            message_dicts, params = self._create_message_dicts(messages, stop)
            params = {
                **params,
                **({"stream": stream} if stream is not None else {}),
                **kwargs,
            }

            answer = await self.llm.async_client.chat_completion(
                messages=message_dicts, **params
            )
            return self._create_chat_result(answer)
        if _is_huggingface_pipeline(self.llm):
            msg = "async generation is not supported with HuggingFacePipeline"
            raise NotImplementedError(msg)
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if _is_huggingface_endpoint(self.llm):
            message_dicts, params = self._create_message_dicts(messages, stop)
            params = {**params, **kwargs, "stream": True}

            default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
            for chunk in self.llm.client.chat_completion(
                messages=message_dicts, **params
            ):
                if len(chunk["choices"]) == 0:
                    continue
                choice = chunk["choices"][0]
                message_chunk = _convert_chunk_to_message_chunk(
                    chunk, default_chunk_class
                )
                generation_info = {}
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                    generation_info["model_name"] = self.model_id
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
                default_chunk_class = message_chunk.__class__
                generation_chunk = ChatGenerationChunk(
                    message=message_chunk, generation_info=generation_info or None
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                    )
                yield generation_chunk
        else:
            llm_input = self._to_chat_prompt(messages)
            stream_iter = self.llm._stream(
                llm_input, stop=stop, run_manager=run_manager, **kwargs
            )
            for chunk in stream_iter:  # chunk is a GenerationChunk
                chat_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=chunk.text),
                    generation_info=chunk.generation_info,
                )
                yield chat_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk

        async for chunk in await self.llm.async_client.chat_completion(
            messages=message_dicts, **params
        ):
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_id
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    def _to_chat_prompt(
        self,
        messages: list[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            msg = "At least one HumanMessage must be provided!"
            raise ValueError(msg)

        if not isinstance(messages[-1], HumanMessage):
            msg = "Last message must be a HumanMessage!"
            raise ValueError(msg)

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            msg = f"Unknown message type: {type(message)}"
            raise ValueError(msg)

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    def _resolve_model_id(self) -> None:
        """Resolve the model_id from the LLM's inference_server_url."""
        from huggingface_hub import list_inference_endpoints  # type: ignore[import]

        if _is_huggingface_hub(self.llm) or (
            hasattr(self.llm, "repo_id") and self.llm.repo_id
        ):
            self.model_id = self.llm.repo_id
            return
        if _is_huggingface_textgen_inference(self.llm):
            endpoint_url: Optional[str] = self.llm.inference_server_url
        if _is_huggingface_pipeline(self.llm):
            from transformers import AutoTokenizer  # type: ignore[import]

            self.model_id = self.model_id or self.llm.model_id
            self.tokenizer = (
                AutoTokenizer.from_pretrained(self.model_id)
                if self.tokenizer is None
                else self.tokenizer
            )
            return
        if _is_huggingface_endpoint(self.llm):
            self.model_id = self.llm.repo_id or self.llm.model
            return
        endpoint_url = self.llm.endpoint_url
        available_endpoints = list_inference_endpoints("*")
        for endpoint in available_endpoints:
            if endpoint.url == endpoint_url:
                self.model_id = endpoint.repository

        if not self.model_id:
            msg = (
                "Failed to resolve model_id:"
                f"Could not find model id for inference server: {endpoint_url}"
                "Make sure that your Hugging Face token has access to the endpoint."
            )
            raise ValueError(msg)

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required"], bool]  # noqa: PYI051
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.

        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                msg = (
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
                raise ValueError(msg)
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]["function"]["name"]
                    != tool_choice["function"]["name"]
                ):
                    msg = (
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[dict, type[BaseModel]]] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a typedDict class (support added in 0.1.7),

                Pydantic class is currently supported.

            method: The method for steering model generation, one of:

                - "function_calling": uses tool-calling features.
                - "json_schema": uses dedicated structured output features.
                - "json_mode": uses JSON mode.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

            kwargs:
                Additional parameters to pass to the underlying LLM's
                :meth:`langchain_core.language_models.chat.BaseChatModel.bind`
                method, such as `response_format` or `ls_structured_output_format`.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs
            an instance of ``schema`` (i.e., a Pydantic object).

            Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:
                - ``"raw"``: BaseMessage
                - ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
                - ``"parsing_error"``: Optional[BaseException]

        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                msg = "Pydantic schema is not supported for function calling"
                raise NotImplementedError(msg)
            output_parser: Union[JsonOutputKeyToolsParser, JsonOutputParser] = (
                JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
            )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_schema = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={"type": "json_object", "schema": formatted_schema},
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": schema,
                },
            )
            output_parser: Union[  # type: ignore[no-redef]
                JsonOutputKeyToolsParser, JsonOutputParser
            ] = JsonOutputParser()  # type: ignore[arg-type]
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            output_parser: Union[  # type: ignore[no-redef]
                JsonOutputKeyToolsParser, JsonOutputParser
            ] = JsonOutputParser()  # type: ignore[arg-type]
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: Optional[list[str]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get default parameters for calling Hugging Face Inference Providers API."""
        params = {
            "model": self.model_id,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            "stop": self.stop,
            **(self.model_kwargs if self.model_kwargs else {}),
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat-wrapper"
