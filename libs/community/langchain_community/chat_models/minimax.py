"""Wrapper around Minimax chat models."""

import json
import logging
from contextlib import asynccontextmanager, contextmanager
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

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
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import get_fields
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)

logger = logging.getLogger(__name__)


@contextmanager
def connect_httpx_sse(client: Any, method: str, url: str, **kwargs: Any) -> Iterator:
    """Context manager for connecting to an SSE stream.

    Args:
        client: The httpx client.
        method: The HTTP method.
        url: The URL to connect to.
        kwargs: Additional keyword arguments to pass to the client.

    Yields:
        An EventSource object.
    """
    from httpx_sse import EventSource

    with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


@asynccontextmanager
async def aconnect_httpx_sse(
    client: Any, method: str, url: str, **kwargs: Any
) -> AsyncIterator:
    """Async context manager for connecting to an SSE stream.

    Args:
        client: The httpx client.
        method: The HTTP method.
        url: The URL to connect to.
        kwargs: Additional keyword arguments to pass to the client.

    Yields:
        An EventSource object.
    """
    from httpx_sse import EventSource

    async with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain messages to Dict."""
    message_dict: Dict[str, Any]
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.additional_kwargs.get("tool_calls"),
        }
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
            "name": message.name or message.additional_kwargs.get("name"),
        }
    else:
        raise TypeError(f"Got unknown type '{message.__class__.__name__}'.")
    return message_dict


def _convert_dict_to_message(dct: Dict[str, Any]) -> BaseMessage:
    """Convert a dict to LangChain message."""
    role = dct.get("role")
    content = dct.get("content", "")
    if role == "assistant":
        additional_kwargs = {}
        tool_calls = dct.get("tool_calls", None)
        if tool_calls is not None:
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def _convert_delta_to_message_chunk(
    dct: Dict[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = dct.get("role")
    content = dct.get("content", "")
    additional_kwargs = {}
    tool_calls = dct.get("tool_call", None)
    if tool_calls is not None:
        additional_kwargs["tool_calls"] = tool_calls

    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


class MiniMaxChat(BaseChatModel):
    """MiniMax chat model integration.

    Setup:
        To use, you should have the environment variable``MINIMAX_API_KEY`` set with
    your API KEY.

        .. code-block:: bash

            export MINIMAX_API_KEY="your-api-key"

    Key init args — completion params:
        model: Optional[str]
            Name of MiniMax model to use.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        temperature: Optional[float]
            Sampling temperature.
        top_p: Optional[float]
            Total probability mass of tokens to consider at each step.
        streaming: Optional[bool]
             Whether to stream the results or not.

    Key init args — client params:
        api_key: Optional[str]
            MiniMax API key. If not passed in will be read from env var MINIMAX_API_KEY.
        base_url: Optional[str]
            Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import MiniMaxChat

            chat = MiniMaxChat(
                api_key=api_key,
                model='abab6.5-chat',
                # temperature=...,
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
                ("human", "我喜欢编程。"),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='I enjoy programming.',
                response_metadata={
                    'token_usage': {'total_tokens': 48},
                    'model_name': 'abab6.5-chat',
                    'finish_reason': 'stop'
                },
                id='run-42d62ba6-5dc1-4e16-98dc-f72708a4162d-0'
            )

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='I' id='run-a5837c45-4aaa-4f64-9ab4-2679bbd55522'
            content=' enjoy programming.' response_metadata={'finish_reason': 'stop'} id='run-a5837c45-4aaa-4f64-9ab4-2679bbd55522'

        .. code-block:: python

            stream = chat.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content='I enjoy programming.',
                response_metadata={'finish_reason': 'stop'},
                id='run-01aed0a0-61c4-4709-be22-c6d8b17155d6'
            )

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

            # stream
            # async for chunk in chat.astream(messages):
            #     print(chunk)

            # batch
            # await chat.abatch([messages])

        .. code-block:: python

            AIMessage(
                content='I enjoy programming.',
                response_metadata={
                    'token_usage': {'total_tokens': 48},
                    'model_name': 'abab6.5-chat',
                    'finish_reason': 'stop'
                },
                id='run-c263b6f1-1736-4ece-a895-055c26b3436f-0'
            )

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )


            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )

            chat_with_tools = chat.bind_tools([GetWeather, GetPopulation])
            ai_msg = chat_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python

            [
                {
                    'name': 'GetWeather',
                    'args': {'location': 'LA'},
                    'id': 'call_function_2140449382',
                    'type': 'tool_call'
                }
            ]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


            structured_chat = chat.with_structured_output(Joke)
            structured_chat.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(
                setup='Why do cats have nine lives?',
                punchline='Because they are so cute and cuddly!',
                rating=None
            )

    Response metadata
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'token_usage': {'total_tokens': 48},
             'model_name': 'abab6.5-chat',
             'finish_reason': 'stop'}

    """  # noqa: E501

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "minimax"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs,
        }

    _client: Any = None
    model: str = "abab6.5-chat"
    """Model name to use."""
    max_tokens: int = 256
    """Denotes the number of tokens to predict per generation."""
    temperature: float = 0.7
    """A non-negative float that tunes the degree of randomness in generation."""
    top_p: float = 0.95
    """Total probability mass of tokens to consider at each step."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    minimax_api_host: str = Field(
        default="https://api.minimax.chat/v1/text/chatcompletion_v2", alias="base_url"
    )
    minimax_group_id: Optional[str] = Field(default=None, alias="group_id")
    """[DEPRECATED, keeping it for for backward compatibility] Group Id"""
    minimax_api_key: SecretStr = Field(alias="api_key")
    """Minimax API Key"""
    streaming: bool = False
    """Whether to stream the results or not."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        values["minimax_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                ["minimax_api_key", "api_key"],
                "MINIMAX_API_KEY",
            )
        )

        default_values = {
            name: field.default
            for name, field in get_fields(cls).items()
            if field.default is not None
        }
        default_values.update(values)

        # Get custom api url from environment.
        values["minimax_api_host"] = get_from_dict_or_env(
            values,
            ["minimax_api_host", "base_url"],
            "MINIMAX_API_HOST",
            default_values["minimax_api_host"],
        )
        return values

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_payload_parameters(  # type: ignore[no-untyped-def]
        self, messages: List[BaseMessage], is_stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Create API request body parameters."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        payload = self._default_params
        payload["messages"] = message_dicts

        self._reformat_function_parameters(kwargs.get("tools", {}))
        payload.update(**kwargs)

        if is_stream:
            payload["stream"] = True

        return payload

    @staticmethod
    def _reformat_function_parameters(tools_arg: Dict[Any, Any]) -> None:
        """Reformat the function parameters to strings."""
        for tool_arg in tools_arg:
            if tool_arg["type"] == "function" and not isinstance(
                tool_arg["function"]["parameters"], str
            ):
                tool_arg["function"]["parameters"] = json.dumps(
                    tool_arg["function"]["parameters"]
                )

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
            stream: Whether to stream the results or not.

        Returns:
            The ChatResult that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        if not messages:
            raise ValueError(
                "You should provide at least one message to start the chat!"
            )
        is_stream = stream if stream is not None else self.streaming
        if is_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        payload = self._create_payload_parameters(messages, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            response = client.post(self.minimax_api_host, json=payload)
            response.raise_for_status()

        return self._create_chat_result(response.json())

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response in chunks."""
        payload = self._create_payload_parameters(messages, is_stream=True, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            with connect_httpx_sse(
                client, "POST", self.minimax_api_host, json=payload
            ) as event_source:
                for sse in event_source.iter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], AIMessageChunk
                    )
                    finish_reason = choice.get("finish_reason", None)

                    generation_info = (
                        {"finish_reason": finish_reason}
                        if finish_reason is not None
                        else None
                    )
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

                    if finish_reason is not None:
                        break

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not messages:
            raise ValueError(
                "You should provide at least one message to start the chat!"
            )
        is_stream = stream if stream is not None else self.streaming
        if is_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        payload = self._create_payload_parameters(messages, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            response = await client.post(self.minimax_api_host, json=payload)
            response.raise_for_status()
        return self._create_chat_result(response.json())

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._create_payload_parameters(messages, is_stream=True, **kwargs)
        api_key = ""
        if self.minimax_api_key is not None:
            api_key = self.minimax_api_key.get_secret_value()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async with aconnect_httpx_sse(
                client, "POST", self.minimax_api_host, json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], AIMessageChunk
                    )
                    finish_reason = choice.get("finish_reason", None)

                    generation_info = (
                        {"finish_reason": finish_reason}
                        if finish_reason is not None
                        else None
                    )
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info
                    )
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

                    if finish_reason is not None:
                        break

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class: `~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

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
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_community.chat_models import MiniMaxChat
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = MiniMaxChat()
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='A pound of bricks and a pound of feathers weigh the same.',
                #     justification='The weight of the feathers is much less dense than the weight of the bricks, but since both weigh one pound, they weigh the same.'
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_community.chat_models import MiniMaxChat
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = MiniMaxChat()
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_function_8953642285', 'type': 'function', 'function': {'name': 'AnswerWithJustification', 'arguments': '{"answer": "A pound of bricks and a pound of feathers weigh the same.", "justification": "The weight of the feathers is much less dense than the weight of the bricks, but since both weigh one pound, they weigh the same."}'}}]}, response_metadata={'token_usage': {'total_tokens': 257}, 'model_name': 'abab6.5-chat', 'finish_reason': 'tool_calls'}, id='run-d897e037-2796-49f5-847e-f9f69dd390db-0', tool_calls=[{'name': 'AnswerWithJustification', 'args': {'answer': 'A pound of bricks and a pound of feathers weigh the same.', 'justification': 'The weight of the feathers is much less dense than the weight of the bricks, but since both weigh one pound, they weigh the same.'}, 'id': 'call_function_8953642285', 'type': 'tool_call'}]),
                #     'parsed': AnswerWithJustification(answer='A pound of bricks and a pound of feathers weigh the same.', justification='The weight of the feathers is much less dense than the weight of the bricks, but since both weigh one pound, they weigh the same.'),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_community.chat_models import MiniMaxChat
                from pydantic import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = MiniMaxChat()
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> {
                #     'answer': 'A pound of bricks and a pound of feathers both weigh the same, which is a pound.',
                #     'justification': 'The difference is that bricks are much denser than feathers, so a pound of bricks will take up much less space than a pound of feathers.'
                # }
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        llm = self.bind_tools([schema])
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema],  # type: ignore[list-item]
                first_tool_only=True,  # type: ignore[list-item]
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
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
