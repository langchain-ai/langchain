import json
import os
import re
import warnings
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

import anthropic
from langchain_core._api import beta, deprecated
from langchain_core.callbacks import (
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
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    build_extra_kwargs,
    convert_to_secret_str,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_anthropic.output_parsers import ToolsOutputParser, extract_tool_calls

_message_type_lookups = {
    "human": "user",
    "ai": "assistant",
    "AIMessageChunk": "assistant",
    "HumanMessageChunk": "user",
}


def _format_image(image_url: str) -> Dict:
    """
    Formats an image of format data:image/jpeg;base64,{b64_string}
    to a dict for anthropic api

    {
      "type": "base64",
      "media_type": "image/jpeg",
      "data": "/9j/4AAQSkZJRg...",
    }

    And throws an error if it's not a b64 image
    """
    regex = r"^data:(?P<media_type>image/.+);base64,(?P<data>.+)$"
    match = re.match(regex, image_url)
    if match is None:
        raise ValueError(
            "Anthropic only supports base64-encoded images currently."
            " Example: data:image/png;base64,'/9j/4AAQSk'..."
        )
    return {
        "type": "base64",
        "media_type": match.group("media_type"),
        "data": match.group("data"),
    }


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:
    """Merge runs of human/tool messages into single human messages with content blocks."""  # noqa: E501
    merged: list = []
    for curr in messages:
        curr = curr.copy(deep=True)
        if isinstance(curr, ToolMessage):
            if isinstance(curr.content, str):
                curr = HumanMessage(
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                        }
                    ]
                )
            else:
                curr = HumanMessage(curr.content)
        last = merged[-1] if merged else None
        if isinstance(last, HumanMessage) and isinstance(curr, HumanMessage):
            if isinstance(last.content, str):
                new_content: List = [{"type": "text", "text": last.content}]
            else:
                new_content = last.content
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            last.content = new_content
        else:
            merged.append(curr)
    return merged


def _format_messages(messages: List[BaseMessage]) -> Tuple[Optional[str], List[Dict]]:
    """Format messages for anthropic."""

    """
    [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_AnthropicMessageContent(text=m.content).dict()],
                }
                for m in messages
            ]
    """
    system: Optional[str] = None
    formatted_messages: List[Dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            if not isinstance(message.content, str):
                raise ValueError(
                    "System message must be a string, "
                    f"instead was: {type(message.content)}"
                )
            system = message.content
            continue

        role = _message_type_lookups[message.type]
        content: Union[str, List]

        if not isinstance(message.content, str):
            # parse as dict
            assert isinstance(
                message.content, list
            ), "Anthropic message content must be str or list of dicts"

            # populate content
            content = []
            for item in message.content:
                if isinstance(item, str):
                    content.append(
                        {
                            "type": "text",
                            "text": item,
                        }
                    )
                elif isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError("Dict content item must have a type key")
                    elif item["type"] == "image_url":
                        # convert format
                        source = _format_image(item["image_url"]["url"])
                        content.append(
                            {
                                "type": "image",
                                "source": source,
                            }
                        )
                    elif item["type"] == "tool_use":
                        item.pop("text", None)
                        content.append(item)
                    elif item["type"] == "text":
                        text = item.get("text", "")
                        # Only add non-empty strings for now as empty ones are not
                        # accepted.
                        # https://github.com/anthropics/anthropic-sdk-python/issues/461
                        if text.strip():
                            content.append(
                                {
                                    "type": "text",
                                    "text": text,
                                }
                            )
                    else:
                        content.append(item)
                else:
                    raise ValueError(
                        f"Content items must be str or dict, instead was: {type(item)}"
                    )
        elif (
            isinstance(message, AIMessage)
            and not isinstance(message.content, list)
            and message.tool_calls
        ):
            content = (
                []
                if not message.content
                else [{"type": "text", "text": message.content}]
            )
            # Note: Anthropic can't have invalid tool calls as presently defined,
            # since the model already returns dicts args not JSON strings, and invalid
            # tool calls are those with invalid JSON for args.
            content += _lc_tool_calls_to_anthropic_tool_use_blocks(message.tool_calls)
        else:
            content = message.content

        formatted_messages.append(
            {
                "role": role,
                "content": content,
            }
        )
    return system, formatted_messages


class ChatAnthropic(BaseChatModel):
    """Anthropic chat model.

    To use, you should have the environment variable ``ANTHROPIC_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(model='claude-3-opus-20240229')
    """

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    _client: anthropic.Client = Field(default=None)
    _async_client: anthropic.AsyncClient = Field(default=None)

    model: str = Field(alias="model_name")
    """Model name to use."""

    max_tokens: int = Field(default=1024, alias="max_tokens_to_sample")
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    default_request_timeout: Optional[float] = Field(None, alias="timeout")
    """Timeout for requests to Anthropic Completion API."""

    # sdk default = 2: https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#retries
    max_retries: int = 2
    """Number of retries allowed for requests sent to the Anthropic Completion API."""

    anthropic_api_url: Optional[str] = None

    anthropic_api_key: Optional[SecretStr] = Field(None, alias="api_key")
    """Automatically read from env var `ANTHROPIC_API_KEY` if not provided."""

    default_headers: Optional[Mapping[str, str]] = None
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    streaming: bool = False
    """Whether to use streaming or not."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"anthropic_api_key": "ANTHROPIC_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "anthropic"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "model_kwargs": self.model_kwargs,
            "streaming": self.streaming,
            "max_retries": self.max_retries,
            "default_request_timeout": self.default_request_timeout,
        }

    @root_validator(pre=True)
    def build_extra(cls, values: Dict) -> Dict:
        extra = values.get("model_kwargs", {})
        all_required_field_names = get_pydantic_field_names(cls)
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        anthropic_api_key = convert_to_secret_str(
            values.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY") or ""
        )
        values["anthropic_api_key"] = anthropic_api_key
        api_key = anthropic_api_key.get_secret_value()
        api_url = (
            values.get("anthropic_api_url")
            or os.environ.get("ANTHROPIC_API_URL")
            or "https://api.anthropic.com"
        )
        values["anthropic_api_url"] = api_url
        client_params = {
            "api_key": api_key,
            "base_url": api_url,
            "max_retries": values["max_retries"],
            "default_headers": values.get("default_headers"),
        }
        # value <= 0 indicates the param should be ignored. None is a meaningful value
        # for Anthropic client and treated differently than not specifying the param at
        # all.
        if (
            values["default_request_timeout"] is None
            or values["default_request_timeout"] > 0
        ):
            client_params["timeout"] = values["default_request_timeout"]

        values["_client"] = anthropic.Client(**client_params)
        values["_async_client"] = anthropic.AsyncClient(**client_params)
        return values

    def _format_params(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> Dict:
        # get system prompt if any
        system, formatted_messages = _format_messages(messages)
        rtn = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": stop,
            "system": system,
            **self.model_kwargs,
            **kwargs,
        }
        rtn = {k: v for k, v in rtn.items() if v is not None}

        return rtn

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if _tools_in_params(params):
            result = self._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            message = result.generations[0].message
            if isinstance(message, AIMessage) and message.tool_calls is not None:
                tool_call_chunks = [
                    {
                        "name": tool_call["name"],
                        "args": json.dumps(tool_call["args"]),
                        "id": tool_call["id"],
                        "index": idx,
                    }
                    for idx, tool_call in enumerate(message.tool_calls)
                ]
                message_chunk = AIMessageChunk(
                    content=message.content,
                    tool_call_chunks=tool_call_chunks,
                )
                yield ChatGenerationChunk(message=message_chunk)
            else:
                yield cast(ChatGenerationChunk, result.generations[0])
            return
        with self._client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if _tools_in_params(params):
            warnings.warn("stream: Tool use is not yet supported in streaming mode.")
            result = await self._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            message = result.generations[0].message
            if isinstance(message, AIMessage) and message.tool_calls is not None:
                tool_call_chunks = [
                    {
                        "name": tool_call["name"],
                        "args": json.dumps(tool_call["args"]),
                        "id": tool_call["id"],
                        "index": idx,
                    }
                    for idx, tool_call in enumerate(message.tool_calls)
                ]
                message_chunk = AIMessageChunk(
                    content=message.content,
                    tool_call_chunks=tool_call_chunks,
                )
                yield ChatGenerationChunk(message=message_chunk)
            else:
                yield cast(ChatGenerationChunk, result.generations[0])
            return
        async with self._async_client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    await run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        data_dict = data.model_dump()
        content = data_dict["content"]
        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        if len(content) == 1 and content[0]["type"] == "text":
            msg = AIMessage(content=content[0]["text"])
        elif any(block["type"] == "tool_use" for block in content):
            tool_calls = extract_tool_calls(content)
            msg = AIMessage(
                content=content,
                tool_calls=tool_calls,
            )
        else:
            msg = AIMessage(content=content)
        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output=llm_output,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            if _tools_in_params(params):
                warnings.warn(
                    "stream: Tool use is not yet supported in streaming mode."
                )
            else:
                stream_iter = self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return generate_from_stream(stream_iter)
        if _tools_in_params(params):
            data = self._client.beta.tools.messages.create(**params)
        else:
            data = self._client.messages.create(**params)
        return self._format_output(data, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            if _tools_in_params(params):
                warnings.warn(
                    "stream: Tool use is not yet supported in streaming mode."
                )
            else:
                stream_iter = self._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return await agenerate_from_stream(stream_iter)
        if _tools_in_params(params):
            data = await self._async_client.beta.tools.messages.create(**params)
        else:
            data = await self._async_client.messages.create(**params)
        return self._format_output(data, **kwargs)

    @beta()
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to bind.

        Example:
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from langchain_core.pydantic_v1 import BaseModel, Field

                class GetWeather(BaseModel):
                    '''Get the current weather in a given location'''

                    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


                llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
                llm_with_tools = llm.bind_tools([GetWeather])
                llm_with_tools.invoke("what is the weather like in San Francisco",)
                # -> AIMessage(
                #     content=[
                #         {'text': '<thinking>\nBased on the user\'s question, the relevant function to call is GetWeather, which requires the "location" parameter.\n\nThe user has directly specified the location as "San Francisco". Since San Francisco is a well known city, I can reasonably infer they mean San Francisco, CA without needing the state specified.\n\nAll the required parameters are provided, so I can proceed with the API call.\n</thinking>', 'type': 'text'},
                #         {'text': None, 'type': 'tool_use', 'id': 'toolu_01SCgExKzQ7eqSkMHfygvYuu', 'name': 'GetWeather', 'input': {'location': 'San Francisco, CA'}}
                #     ],
                #     response_metadata={'id': 'msg_01GM3zQtoFv8jGQMW7abLnhi', 'model': 'claude-3-opus-20240229', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 487, 'output_tokens': 145}},
                #     id='run-87b1331e-9251-4a68-acef-f0a018b639cc-0'
                # )
        """  # noqa: E501
        formatted_tools = [convert_to_anthropic_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, **kwargs)

    @beta()
    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input. The output type depends on
            include_raw and schema.

            If include_raw is True then output is a dict with keys:
                raw: BaseMessage,
                parsed: Optional[_DictOrPydantic],
                parsing_error: Optional[BaseException],

            If include_raw is False and schema is a Dict then the runnable outputs a Dict.
            If include_raw is False and schema is a Type[BaseModel] then the runnable
            outputs a BaseModel.

        Example: Pydantic schema (include_raw=False):
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example:  Pydantic schema (include_raw=True):
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Dict schema (include_raw=False):
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic

                schema = {
                    "name": "AnswerWithJustification",
                    "description": "An answer to the user question along with justification for the answer.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "justification": {"type": "string"},
                        },
                        "required": ["answer", "justification"]
                    }
                }
                llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
                structured_llm = llm.with_structured_output(schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        """  # noqa: E501
        llm = self.bind_tools([schema])
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            output_parser = ToolsOutputParser(
                first_tool_only=True, pydantic_schemas=[schema]
            )
        else:
            output_parser = ToolsOutputParser(first_tool_only=True, args_only=True)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser


class AnthropicTool(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]


def convert_to_anthropic_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> AnthropicTool:
    # already in Anthropic tool format
    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
        return AnthropicTool(tool)  # type: ignore
    else:
        formatted = convert_to_openai_tool(tool)["function"]
        return AnthropicTool(
            name=formatted["name"],
            description=formatted["description"],
            input_schema=formatted["parameters"],
        )


def _tools_in_params(params: dict) -> bool:
    return "tools" in params or (
        "extra_body" in params and params["extra_body"].get("tools")
    )


class _AnthropicToolUse(TypedDict):
    type: Literal["tool_use"]
    name: str
    input: dict
    id: str


def _lc_tool_calls_to_anthropic_tool_use_blocks(
    tool_calls: List[ToolCall],
) -> List[_AnthropicToolUse]:
    blocks = []
    for tool_call in tool_calls:
        blocks.append(
            _AnthropicToolUse(
                type="tool_use",
                name=tool_call["name"],
                input=tool_call["args"],
                id=cast(str, tool_call["id"]),
            )
        )
    return blocks


@deprecated(since="0.1.0", removal="0.2.0", alternative="ChatAnthropic")
class ChatAnthropicMessages(ChatAnthropic):
    pass
