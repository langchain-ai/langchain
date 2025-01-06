import copy
import re
import warnings
from functools import cached_property
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
    LangSmithParams,
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
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    from_env,
    get_pydantic_field_names,
    secret_from_env,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import NotRequired, TypedDict

from langchain_anthropic.output_parsers import extract_tool_calls

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
        if isinstance(curr, ToolMessage):
            if (
                isinstance(curr.content, list)
                and curr.content
                and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in curr.content
                )
            ):
                curr = HumanMessage(curr.content)  # type: ignore[misc]
            else:
                curr = HumanMessage(  # type: ignore[misc]
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                            "is_error": curr.status == "error",
                        }
                    ]
                )
        last = merged[-1] if merged else None
        if any(
            all(isinstance(m, c) for m in (curr, last))
            for c in (SystemMessage, HumanMessage)
        ):
            if isinstance(cast(BaseMessage, last).content, str):
                new_content: List = [
                    {"type": "text", "text": cast(BaseMessage, last).content}
                ]
            else:
                new_content = copy.copy(cast(list, cast(BaseMessage, last).content))
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            merged[-1] = curr.model_copy(update={"content": new_content})
        else:
            merged.append(curr)
    return merged


def _format_messages(
    messages: List[BaseMessage],
) -> Tuple[Union[str, List[Dict], None], List[Dict]]:
    """Format messages for anthropic."""

    """
    [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_AnthropicMessageContent(text=m.content).model_dump()],
                }
                for m in messages
            ]
    """
    system: Union[str, List[Dict], None] = None
    formatted_messages: List[Dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if system is not None:
                raise ValueError("Received multiple non-consecutive system messages.")
            elif isinstance(message.content, list):
                system = [
                    (
                        block
                        if isinstance(block, dict)
                        else {"type": "text", "text": block}
                    )
                    for block in message.content
                ]
            else:
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
            for block in message.content:
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                elif isinstance(block, dict):
                    if "type" not in block:
                        raise ValueError("Dict content block must have a type key")
                    elif block["type"] == "image_url":
                        # convert format
                        source = _format_image(block["image_url"]["url"])
                        content.append({"type": "image", "source": source})
                    elif block["type"] == "tool_use":
                        # If a tool_call with the same id as a tool_use content block
                        # exists, the tool_call is preferred.
                        if isinstance(message, AIMessage) and block["id"] in [
                            tc["id"] for tc in message.tool_calls
                        ]:
                            overlapping = [
                                tc
                                for tc in message.tool_calls
                                if tc["id"] == block["id"]
                            ]
                            content.extend(
                                _lc_tool_calls_to_anthropic_tool_use_blocks(overlapping)
                            )
                        else:
                            block.pop("text", None)
                            content.append(block)
                    elif block["type"] == "text":
                        text = block.get("text", "")
                        # Only add non-empty strings for now as empty ones are not
                        # accepted.
                        # https://github.com/anthropics/anthropic-sdk-python/issues/461
                        if text.strip():
                            content.append(
                                {
                                    k: v
                                    for k, v in block.items()
                                    if k in ("type", "text", "cache_control")
                                }
                            )
                    elif block["type"] == "tool_result":
                        tool_content = _format_messages(
                            [HumanMessage(block["content"])]
                        )[1][0]["content"]
                        content.append({**block, **{"content": tool_content}})
                    else:
                        content.append(block)
                else:
                    raise ValueError(
                        f"Content blocks must be str or dict, instead was: "
                        f"{type(block)}"
                    )
        else:
            content = message.content

        # Ensure all tool_calls have a tool_use content block
        if isinstance(message, AIMessage) and message.tool_calls:
            content = content or []
            content = (
                [{"type": "text", "text": message.content}]
                if isinstance(content, str) and content
                else content
            )
            tool_use_ids = [
                cast(dict, block)["id"]
                for block in content
                if cast(dict, block)["type"] == "tool_use"
            ]
            missing_tool_calls = [
                tc for tc in message.tool_calls if tc["id"] not in tool_use_ids
            ]
            cast(list, content).extend(
                _lc_tool_calls_to_anthropic_tool_use_blocks(missing_tool_calls)
            )

        formatted_messages.append({"role": role, "content": content})
    return system, formatted_messages


class ChatAnthropic(BaseChatModel):
    """Anthropic chat models.

    See https://docs.anthropic.com/en/docs/models-overview for a list of the latest models.

    Setup:
        Install ``langchain-anthropic`` and set environment variable ``ANTHROPIC_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-anthropic
            export ANTHROPIC_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Anthropic model to use. E.g. "claude-3-sonnet-20240229".
        temperature: float
            Sampling temperature. Ranges from 0.0 to 1.0.
        max_tokens: int
            Max number of tokens to generate.

    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries if a request fails.
        api_key: Optional[str]
            Anthropic API key. If not passed in will be read from env var ANTHROPIC_API_KEY.
        base_url: Optional[str]
            Base URL for API requests. Only specify if using a proxy or service
            emulator.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0,
                max_tokens=1024,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # base_url="...",
                # other params...
            )

    **NOTE**: Any param which is not explicitly supported will be passed directly to the
    ``anthropic.Anthropic.messages.create(...)`` API every time to the model is
    invoked. For example:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic
            import anthropic

            ChatAnthropic(..., extra_headers={}).invoke(...)

            # results in underlying API call of:

            anthropic.Anthropic(..).messages.create(..., extra_headers={})

            # which is also equivalent to:

            ChatAnthropic(...).invoke(..., extra_headers={})

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content="J'aime la programmation.", response_metadata={'id': 'msg_01Trik66aiQ9Z1higrD5XFx3', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 25, 'output_tokens': 11}}, id='run-5886ac5f-3c2e-49f5-8a44-b1e92808c929-0', usage_metadata={'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content='J', id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')
            AIMessageChunk(content="'", id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')
            AIMessageChunk(content='a', id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')
            AIMessageChunk(content='ime', id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')
            AIMessageChunk(content=' la', id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')
            AIMessageChunk(content=' programm', id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')
            AIMessageChunk(content='ation', id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')
            AIMessageChunk(content='.', id='run-272ff5f9-8485-402c-b90d-eac8babc5b25')

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content="J'aime la programmation.", id='run-b34faef0-882f-4869-a19c-ed2b856e6361')

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(content="J'aime la programmation.", response_metadata={'id': 'msg_01Trik66aiQ9Z1higrD5XFx3', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 25, 'output_tokens': 11}}, id='run-5886ac5f-3c2e-49f5-8a44-b1e92808c929-0', usage_metadata={'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36})

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetWeather',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'toolu_01KzpPEAgzura7hpBqwHbWdo'},
             {'name': 'GetWeather',
              'args': {'location': 'New York, NY'},
              'id': 'toolu_01JtgbVGVJbiSwtZk3Uycezx'},
             {'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'toolu_01429aygngesudV9nTbCKGuw'},
             {'name': 'GetPopulation',
              'args': {'location': 'New York, NY'},
              'id': 'toolu_01JPktyd44tVMeBcPPnFSEJG'}]

        See ``ChatAnthropic.bind_tools()`` method for more.

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup='Why was the cat sitting on the computer?', punchline='To keep an eye on the mouse!', rating=None)

        See ``ChatAnthropic.with_structured_output()`` for more.

    Image input:
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            "The image depicts a sunny day with a partly cloudy sky. The sky is a brilliant blue color with scattered white clouds drifting across. The lighting and cloud patterns suggest pleasant, mild weather conditions. The scene shows a grassy field or meadow with a wooden boardwalk trail leading through it, indicating an outdoor setting on a nice day well-suited for enjoying nature."

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36}

        Message chunks containing token usage will be included during streaming by
        default:

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full.usage_metadata

        .. code-block:: python

            {'input_tokens': 25, 'output_tokens': 11, 'total_tokens': 36}

        These can be disabled by setting ``stream_usage=False`` in the stream method,
        or by setting ``stream_usage=False`` when initializing ChatAnthropic.

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'id': 'msg_013xU6FHEGEq76aP4RgFerVT',
             'model': 'claude-3-sonnet-20240229',
             'stop_reason': 'end_turn',
             'stop_sequence': None,
             'usage': {'input_tokens': 25, 'output_tokens': 11}}

    """  # noqa: E501

    model_config = ConfigDict(
        populate_by_name=True,
    )

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

    stop_sequences: Optional[List[str]] = Field(None, alias="stop")
    """Default stop sequences."""

    anthropic_api_url: Optional[str] = Field(
        alias="base_url",
        default_factory=from_env(
            ["ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"],
            default="https://api.anthropic.com",
        ),
    )
    """Base URL for API requests. Only specify if using a proxy or service emulator.

    If a value isn't passed in, will attempt to read the value first from
    ANTHROPIC_API_URL and if that is not set, ANTHROPIC_BASE_URL.
    If neither are set, the default value of 'https://api.anthropic.com' will
    be used.
    """

    anthropic_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("ANTHROPIC_API_KEY", default=""),
    )

    """Automatically read from env var `ANTHROPIC_API_KEY` if not provided."""

    default_headers: Optional[Mapping[str, str]] = None
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    streaming: bool = False
    """Whether to use streaming or not."""

    stream_usage: bool = True
    """Whether to include usage metadata in streaming output. If True, additional
    message chunks will be generated during the stream including usage metadata.
    """

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

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="anthropic",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict) -> Any:
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values

    @cached_property
    def _client_params(self) -> Dict[str, Any]:
        client_params: Dict[str, Any] = {
            "api_key": self.anthropic_api_key.get_secret_value(),
            "base_url": self.anthropic_api_url,
            "max_retries": self.max_retries,
            "default_headers": (self.default_headers or None),
        }
        # value <= 0 indicates the param should be ignored. None is a meaningful value
        # for Anthropic client and treated differently than not specifying the param at
        # all.
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_params["timeout"] = self.default_request_timeout

        return client_params

    @cached_property
    def _client(self) -> anthropic.Client:
        return anthropic.Client(**self._client_params)

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        return anthropic.AsyncClient(**self._client_params)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> Dict:
        messages = self._convert_input(input_).to_messages()
        system, formatted_messages = _format_messages(messages)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": stop or self.stop_sequences,
            "system": system,
            **self.model_kwargs,
            **kwargs,
        }
        return {k: v for k, v in payload.items() if v is not None}

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        stream = self._client.messages.create(**payload)
        coerce_content_to_string = not _tools_in_params(payload)
        for event in stream:
            msg = _make_message_chunk_from_anthropic_event(
                event,
                stream_usage=stream_usage,
                coerce_content_to_string=coerce_content_to_string,
            )
            if msg is not None:
                chunk = ChatGenerationChunk(message=msg)
                if run_manager and isinstance(msg.content, str):
                    run_manager.on_llm_new_token(msg.content, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        stream = await self._async_client.messages.create(**payload)
        coerce_content_to_string = not _tools_in_params(payload)
        async for event in stream:
            msg = _make_message_chunk_from_anthropic_event(
                event,
                stream_usage=stream_usage,
                coerce_content_to_string=coerce_content_to_string,
            )
            if msg is not None:
                chunk = ChatGenerationChunk(message=msg)
                if run_manager and isinstance(msg.content, str):
                    await run_manager.on_llm_new_token(msg.content, chunk=chunk)
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
        msg.usage_metadata = _create_usage_metadata(data.usage)
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
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        data = self._client.messages.create(**payload)
        return self._format_output(data, **kwargs)

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
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        data = await self._async_client.messages.create(**payload)
        return self._format_output(data, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[Dict[str, str], Literal["any", "auto"], str]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        r"""Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports Anthropic format tool schemas and any tool definition handled
                by :meth:`~langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call. Options are:

                - name of the tool as a string or as dict ``{"type": "tool", "name": "<<tool_name>>"}``: calls corresponding tool;
                - ``"auto"``, ``{"type: "auto"}``, or None: automatically selects a tool (including no tool);
                - ``"any"`` or ``{"type: "any"}``: force at least one tool to be called;
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_anthropic.chat_models.ChatAnthropic.bind`.

        Example:
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from pydantic import BaseModel, Field

                class GetWeather(BaseModel):
                    '''Get the current weather in a given location'''

                    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

                class GetPrice(BaseModel):
                    '''Get the price of a specific product.'''

                    product: str = Field(..., description="The product to look up.")


                llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
                llm_with_tools = llm.bind_tools([GetWeather, GetPrice])
                llm_with_tools.invoke("what is the weather like in San Francisco",)
                # -> AIMessage(
                #     content=[
                #         {'text': '<thinking>\nBased on the user\'s question, the relevant function to call is GetWeather, which requires the "location" parameter.\n\nThe user has directly specified the location as "San Francisco". Since San Francisco is a well known city, I can reasonably infer they mean San Francisco, CA without needing the state specified.\n\nAll the required parameters are provided, so I can proceed with the API call.\n</thinking>', 'type': 'text'},
                #         {'text': None, 'type': 'tool_use', 'id': 'toolu_01SCgExKzQ7eqSkMHfygvYuu', 'name': 'GetWeather', 'input': {'location': 'San Francisco, CA'}}
                #     ],
                #     response_metadata={'id': 'msg_01GM3zQtoFv8jGQMW7abLnhi', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 487, 'output_tokens': 145}},
                #     id='run-87b1331e-9251-4a68-acef-f0a018b639cc-0'
                # )

        Example — force tool call with tool_choice 'any':
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from pydantic import BaseModel, Field

                class GetWeather(BaseModel):
                    '''Get the current weather in a given location'''

                    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

                class GetPrice(BaseModel):
                    '''Get the price of a specific product.'''

                    product: str = Field(..., description="The product to look up.")


                llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
                llm_with_tools = llm.bind_tools([GetWeather, GetPrice], tool_choice="any")
                llm_with_tools.invoke("what is the weather like in San Francisco",)


        Example — force specific tool call with tool_choice '<name_of_tool>':
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from pydantic import BaseModel, Field

                class GetWeather(BaseModel):
                    '''Get the current weather in a given location'''

                    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

                class GetPrice(BaseModel):
                    '''Get the price of a specific product.'''

                    product: str = Field(..., description="The product to look up.")


                llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
                llm_with_tools = llm.bind_tools([GetWeather, GetPrice], tool_choice="GetWeather")
                llm_with_tools.invoke("what is the weather like in San Francisco",)

        Example — cache specific tools:
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic, convert_to_anthropic_tool
                from pydantic import BaseModel, Field

                class GetWeather(BaseModel):
                    '''Get the current weather in a given location'''

                    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

                class GetPrice(BaseModel):
                    '''Get the price of a specific product.'''

                    product: str = Field(..., description="The product to look up.")

                # We'll convert our pydantic class to the anthropic tool format
                # before passing to bind_tools so that we can set the 'cache_control'
                # field on our tool.
                cached_price_tool = convert_to_anthropic_tool(GetPrice)
                # Currently the only supported "cache_control" value is
                # {"type": "ephemeral"}.
                cached_price_tool["cache_control"] = {"type": "ephemeral"}

                # We need to pass in extra headers to enable use of the beta cache
                # control API.
                llm = ChatAnthropic(
                    model="claude-3-5-sonnet-20240620",
                    temperature=0,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
                )
                llm_with_tools = llm.bind_tools([GetWeather, cached_price_tool])
                llm_with_tools.invoke("what is the weather like in San Francisco",)

            This outputs:

            .. code-block:: python

                AIMessage(content=[{'text': "Certainly! I can help you find out the current weather in San Francisco. To get this information, I'll use the GetWeather function. Let me fetch that data for you right away.", 'type': 'text'}, {'id': 'toolu_01TS5h8LNo7p5imcG7yRiaUM', 'input': {'location': 'San Francisco, CA'}, 'name': 'GetWeather', 'type': 'tool_use'}], response_metadata={'id': 'msg_01Xg7Wr5inFWgBxE5jH9rpRo', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 171, 'output_tokens': 96, 'cache_creation_input_tokens': 1470, 'cache_read_input_tokens': 0}}, id='run-b36a5b54-5d69-470e-a1b0-b932d00b089e-0', tool_calls=[{'name': 'GetWeather', 'args': {'location': 'San Francisco, CA'}, 'id': 'toolu_01TS5h8LNo7p5imcG7yRiaUM', 'type': 'tool_call'}], usage_metadata={'input_tokens': 171, 'output_tokens': 96, 'total_tokens': 267})

            If we invoke the tool again, we can see that the "usage" information in the AIMessage.response_metadata shows that we had a cache hit:

            .. code-block:: python

                AIMessage(content=[{'text': 'To get the current weather in San Francisco, I can use the GetWeather function. Let me check that for you.', 'type': 'text'}, {'id': 'toolu_01HtVtY1qhMFdPprx42qU2eA', 'input': {'location': 'San Francisco, CA'}, 'name': 'GetWeather', 'type': 'tool_use'}], response_metadata={'id': 'msg_016RfWHrRvW6DAGCdwB6Ac64', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 171, 'output_tokens': 82, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 1470}}, id='run-88b1f825-dcb7-4277-ac27-53df55d22001-0', tool_calls=[{'name': 'GetWeather', 'args': {'location': 'San Francisco, CA'}, 'id': 'toolu_01HtVtY1qhMFdPprx42qU2eA', 'type': 'tool_call'}], usage_metadata={'input_tokens': 171, 'output_tokens': 82, 'total_tokens': 253})

        """  # noqa: E501
        formatted_tools = [convert_to_anthropic_tool(tool) for tool in tools]
        if not tool_choice:
            pass
        elif isinstance(tool_choice, dict):
            kwargs["tool_choice"] = tool_choice
        elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
            kwargs["tool_choice"] = {"type": tool_choice}
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        else:
            raise ValueError(
                f"Unrecognized 'tool_choice' type {tool_choice=}. Expected dict, "
                f"str, or None."
            )
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - an Anthropic tool schema,
                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class.

                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`~langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.
            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".
            kwargs: Additional keyword arguments are ignored.

        Returns:
            A Runnable that takes same inputs as a :class:`~langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs
            an instance of ``schema`` (i.e., a Pydantic object).

            Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:
                - ``"raw"``: BaseMessage
                - ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
                - ``"parsing_error"``: Optional[BaseException]

        Example: Pydantic schema (include_raw=False):
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example:  Pydantic schema (include_raw=True):
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
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
                llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
                structured_llm = llm.with_structured_output(schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        .. versionchanged:: 0.1.22

                Added support for TypedDict class as `schema`.

        """  # noqa: E501

        tool_name = convert_to_anthropic_tool(schema)["name"]
        llm = self.bind_tools([schema], tool_choice=tool_name)
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            output_parser = JsonOutputKeyToolsParser(
                key_name=tool_name, first_tool_only=True
            )

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

    @beta()
    def get_num_tokens_from_messages(
        self,
        messages: List[BaseMessage],
        tools: Optional[
            Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]
        ] = None,
    ) -> int:
        """Count tokens in a sequence of input messages.

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of dict, BaseModel, function, or BaseTools
                to be converted to tool schemas.

        Basic usage:
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from langchain_core.messages import HumanMessage, SystemMessage

                llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

                messages = [
                    SystemMessage(content="You are a scientist"),
                    HumanMessage(content="Hello, Claude"),
                ]
                llm.get_num_tokens_from_messages(messages)

            .. code-block:: none

                14

        Pass tool schemas:
            .. code-block:: python

                from langchain_anthropic import ChatAnthropic
                from langchain_core.messages import HumanMessage
                from langchain_core.tools import tool

                llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

                @tool(parse_docstring=True)
                def get_weather(location: str) -> str:
                    \"\"\"Get the current weather in a given location

                    Args:
                        location: The city and state, e.g. San Francisco, CA
                    \"\"\"
                    return "Sunny"

                messages = [
                    HumanMessage(content="What's the weather like in San Francisco?"),
                ]
                llm.get_num_tokens_from_messages(messages, tools=[get_weather])

            .. code-block:: none

                403

        .. versionchanged:: 0.3.0

                Uses Anthropic's token counting API to count tokens in messages. See:
                https://docs.anthropic.com/en/docs/build-with-claude/token-counting
        """
        formatted_system, formatted_messages = _format_messages(messages)
        kwargs: Dict[str, Any] = {}
        if isinstance(formatted_system, str):
            kwargs["system"] = formatted_system
        if tools:
            kwargs["tools"] = [convert_to_anthropic_tool(tool) for tool in tools]

        response = self._client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=self.model,
            messages=formatted_messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.input_tokens


class AnthropicTool(TypedDict):
    """Anthropic tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    cache_control: NotRequired[Dict[str, str]]


def convert_to_anthropic_tool(
    tool: Union[Dict[str, Any], Type, Callable, BaseTool],
) -> AnthropicTool:
    """Convert a tool-like object to an Anthropic tool definition."""
    # already in Anthropic tool format
    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
        anthropic_formatted = AnthropicTool(tool)  # type: ignore
    else:
        oai_formatted = convert_to_openai_tool(tool)["function"]
        anthropic_formatted = AnthropicTool(
            name=oai_formatted["name"],
            description=oai_formatted["description"],
            input_schema=oai_formatted["parameters"],
        )
    return anthropic_formatted


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


def _make_message_chunk_from_anthropic_event(
    event: anthropic.types.RawMessageStreamEvent,
    *,
    stream_usage: bool = True,
    coerce_content_to_string: bool,
) -> Optional[AIMessageChunk]:
    """Convert Anthropic event to AIMessageChunk.

    Note that not all events will result in a message chunk. In these cases
    we return None.
    """
    message_chunk: Optional[AIMessageChunk] = None
    # See https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/streaming/_messages.py  # noqa: E501
    if event.type == "message_start" and stream_usage:
        usage_metadata = _create_usage_metadata(event.message.usage)
        message_chunk = AIMessageChunk(
            content="" if coerce_content_to_string else [],
            usage_metadata=usage_metadata,
        )
    elif (
        event.type == "content_block_start"
        and event.content_block is not None
        and event.content_block.type == "tool_use"
    ):
        if coerce_content_to_string:
            warnings.warn("Received unexpected tool content block.")
        content_block = event.content_block.model_dump()
        content_block["index"] = event.index
        tool_call_chunk = create_tool_call_chunk(
            index=event.index,
            id=event.content_block.id,
            name=event.content_block.name,
            args="",
        )
        message_chunk = AIMessageChunk(
            content=[content_block],
            tool_call_chunks=[tool_call_chunk],  # type: ignore
        )
    elif event.type == "content_block_delta":
        if event.delta.type == "text_delta":
            if coerce_content_to_string:
                text = event.delta.text
                message_chunk = AIMessageChunk(content=text)
            else:
                content_block = event.delta.model_dump()
                content_block["index"] = event.index
                content_block["type"] = "text"
                message_chunk = AIMessageChunk(content=[content_block])
        elif event.delta.type == "input_json_delta":
            content_block = event.delta.model_dump()
            content_block["index"] = event.index
            content_block["type"] = "tool_use"
            tool_call_chunk = create_tool_call_chunk(
                index=event.index,
                id=None,
                name=None,
                args=event.delta.partial_json,
            )
            message_chunk = AIMessageChunk(
                content=[content_block],
                tool_call_chunks=[tool_call_chunk],  # type: ignore
            )
    elif event.type == "message_delta" and stream_usage:
        usage_metadata = _create_usage_metadata(event.usage)
        message_chunk = AIMessageChunk(
            content="",
            usage_metadata=usage_metadata,
            response_metadata={
                "stop_reason": event.delta.stop_reason,
                "stop_sequence": event.delta.stop_sequence,
            },
        )
    else:
        pass

    return message_chunk


@deprecated(since="0.1.0", removal="1.0.0", alternative="ChatAnthropic")
class ChatAnthropicMessages(ChatAnthropic):
    pass


def _create_usage_metadata(anthropic_usage: BaseModel) -> UsageMetadata:
    input_token_details: Dict = {
        "cache_read": getattr(anthropic_usage, "cache_read_input_tokens", None),
        "cache_creation": getattr(anthropic_usage, "cache_creation_input_tokens", None),
    }

    # Anthropic input_tokens exclude cached token counts.
    input_tokens = (
        getattr(anthropic_usage, "input_tokens", 0)
        + (input_token_details["cache_read"] or 0)
        + (input_token_details["cache_creation"] or 0)
    )
    output_tokens = getattr(anthropic_usage, "output_tokens", 0)
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None}
        ),
    )
