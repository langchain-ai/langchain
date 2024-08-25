"""Databricks chat models."""

import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    PrivateAttr,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_databricks.utils import get_deployment_client

logger = logging.getLogger(__name__)


class ChatDatabricks(BaseChatModel):
    """Databricks chat model integration.

    Setup:
        Install ``langchain-databricks``.

        .. code-block:: bash

            pip install -U langchain-databricks

        If you are outside Databricks, set the Databricks workspace hostname and personal access token to environment variables:

        .. code-block:: bash

            export DATABRICKS_HOSTNAME="https://your-databricks-workspace"
            export DATABRICKS_TOKEN="your-personal-access-token"

    Key init args — completion params:
        endpoint: str
            Name of Databricks Model Serving endpoint to query.
        target_uri: str
            The target URI to use. Defaults to ``databricks``.
        temperature: float
            Sampling temperature. Higher values make the model more creative.
        n: Optional[int]
            The number of completion choices to generate.
        stop: Optional[List[str]]
            List of strings to stop generation at.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        extra_params: Optional[Dict[str, Any]]
            Any extra parameters to pass to the endpoint.

    Instantiate:
        .. code-block:: python

            from langchain_databricks import ChatDatabricks
            llm = ChatDatabricks(
                endpoint="databricks-meta-llama-3-1-405b-instruct",
                temperature=0,
                max_tokens=500,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={
                    'prompt_tokens': 32,
                    'completion_tokens': 9,
                    'total_tokens': 41
                },
                id='run-64eebbdd-88a8-4a25-b508-21e9a5f146c5-0'
            )

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            content='J' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content="'" id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='ad' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='ore' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content=' la' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content=' programm' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='ation' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='.' id='run-609b8f47-e580-4691-9ee4-e2109f53155e'
            content='' response_metadata={'finish_reason': 'stop'} id='run-609b8f47-e580-4691-9ee4-e2109f53155e'

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content="J'adore la programmation.",
                response_metadata={
                    'finish_reason': 'stop'
                },
                id='run-4cef851f-6223-424f-ad26-4a54e5852aa5'
            )

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in llm.astream(messages)

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={
                    'prompt_tokens': 32,
                    'completion_tokens': 9,
                    'total_tokens': 41
                },
                id='run-e4bb043e-772b-4e1d-9f98-77ccc00c0271-0'
            )

    Tool calling:
        .. code-block:: python

            from langchain_core.pydantic_v1 import BaseModel, Field

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

            [
                {
                    'name': 'GetWeather',
                    'args': {
                        'location': 'Los Angeles, CA'
                    },
                    'id': 'call_ea0a6004-8e64-4ae8-a192-a40e295bfa24',
                    'type': 'tool_call'
                }
            ]

        To use tool calls, your model endpoint must support ``tools`` parameter. See [Function calling on Databricks](https://python.langchain.com/v0.2/docs/integrations/chat/databricks/#function-calling-on-databricks) for more information.

    """  # noqa: E501

    endpoint: str
    """Name of Databricks Model Serving endpoint to query."""
    target_uri: str = "databricks"
    """The target URI to use. Defaults to ``databricks``."""
    temperature: float = 0.0
    """Sampling temperature. Higher values make the model more creative."""
    n: int = 1
    """The number of completion choices to generate."""
    stop: Optional[List[str]] = None
    """List of strings to stop generation at."""
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    extra_params: dict = Field(default_factory=dict)
    """Any extra parameters to pass to the endpoint."""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = get_deployment_client(self.target_uri)

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "target_uri": self.target_uri,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "extra_params": self.extra_params,
        }
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        data = self._prepare_inputs(messages, stop, **kwargs)
        resp = self._client.predict(endpoint=self.endpoint, inputs=data)
        return self._convert_response_to_chat_result(resp)

    def _prepare_inputs(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "messages": [_convert_message_to_dict(msg) for msg in messages],
            "temperature": self.temperature,
            "n": self.n,
            **self.extra_params,
            **kwargs,
        }
        if stop := self.stop or stop:
            data["stop"] = stop
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens

        return data

    def _convert_response_to_chat_result(
        self, response: Mapping[str, Any]
    ) -> ChatResult:
        generations = [
            ChatGeneration(
                message=_convert_dict_to_message(choice["message"]),
                generation_info=choice.get("usage", {}),
            )
            for choice in response["choices"]
        ]
        usage = response.get("usage", {})
        return ChatResult(generations=generations, llm_output=usage)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        data = self._prepare_inputs(messages, stop, **kwargs)
        first_chunk_role = None
        for chunk in self._client.predict_stream(endpoint=self.endpoint, inputs=data):
            if chunk["choices"]:
                choice = chunk["choices"][0]

                chunk_delta = choice["delta"]
                if first_chunk_role is None:
                    first_chunk_role = chunk_delta.get("role")

                chunk_message = _convert_dict_to_message_chunk(
                    chunk_delta, first_chunk_role
                )

                generation_info = {}
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                if logprobs := choice.get("logprobs"):
                    generation_info["logprobs"] = logprobs

                chunk = ChatGenerationChunk(
                    message=chunk_message, generation_info=generation_info or None
                )

                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text, chunk=chunk, logprobs=logprobs
                    )

                yield chunk
            else:
                # Handle the case where choices are empty if needed
                continue

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Options are:
                name of the tool (str): calls corresponding tool;
                "auto": automatically selects a tool (including no tool);
                "none": model does not generate any tool calls and instead must
                    generate a standard assistant message;
                "required": the model picks the most relevant tool in tools and
                    must generate a tool call;

                or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-databricks"


### Conversion function to convert Pydantic models to dictionaries and vice versa. ###


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict = {"content": message.content}

    # OpenAI supports "name" field in messages.
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    if id := message.id:
        message_dict["id"] = id

    if isinstance(message, ChatMessage):
        return {"role": message.role, **message_dict}
    elif isinstance(message, HumanMessage):
        return {"role": "user", **message_dict}
    elif isinstance(message, AIMessage):
        if tool_calls := _get_tool_calls_from_ai_message(message):
            message_dict["tool_calls"] = tool_calls  # type: ignore[assignment]
            # If tool calls present, content null value should be None not empty string.
            message_dict["content"] = message_dict["content"] or None  # type: ignore[assignment]
        return {"role": "assistant", **message_dict}
    elif isinstance(message, SystemMessage):
        return {"role": "system", **message_dict}
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            **message_dict,
        }
    elif (
        isinstance(message, FunctionMessage)
        or "function_call" in message.additional_kwargs
    ):
        raise ValueError(
            "Function messages are not supported by Databricks. Please"
            " create a feature request at https://github.com/mlflow/mlflow/issues."
        )
    else:
        raise ValueError(f"Got unknown message type: {type(message)}")


def _get_tool_calls_from_ai_message(message: AIMessage) -> List[Dict]:
    tool_calls = [
        {
            "type": "function",
            "id": tc["id"],
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            },
        }
        for tc in message.tool_calls
    ]

    invalid_tool_calls = [
        {
            "type": "function",
            "id": tc["id"],
            "function": {
                "name": tc["name"],
                "arguments": tc["args"],
            },
        }
        for tc in message.invalid_tool_calls
    ]

    if tool_calls or invalid_tool_calls:
        return tool_calls + invalid_tool_calls

    # Get tool calls from additional kwargs if present.
    return [
        {
            k: v
            for k, v in tool_call.items()  # type: ignore[union-attr]
            if k in {"id", "type", "function"}
        }
        for tool_call in message.additional_kwargs.get("tool_calls", [])
    ]


def _convert_dict_to_message(_dict: Dict) -> BaseMessage:
    role = _dict["role"]
    content = _dict.get("content")
    content = content if content is not None else ""

    if role == "user":
        return HumanMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "assistant":
        additional_kwargs: Dict = {}
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            id=_dict.get("id"),
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    else:
        return ChatMessage(content=content, role=role)


def _convert_dict_to_message_chunk(
    _dict: Mapping[str, Any], default_role: str
) -> BaseMessageChunk:
    role = _dict.get("role", default_role)
    content = _dict.get("content")
    content = content if content is not None else ""

    if role == "user":
        return HumanMessageChunk(content=content)
    elif role == "system":
        return SystemMessageChunk(content=content)
    elif role == "tool":
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=_dict.get("id")
        )
    elif role == "assistant":
        additional_kwargs: Dict = {}
        tool_call_chunks = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            try:
                tool_call_chunks = [
                    tool_call_chunk(
                        name=tc["function"].get("name"),
                        args=tc["function"].get("arguments"),
                        id=tc.get("id"),
                        index=tc["index"],
                    )
                    for tc in raw_tool_calls
                ]
            except KeyError:
                pass
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=_dict.get("id"),
            tool_call_chunks=tool_call_chunks,
        )
    else:
        return ChatMessageChunk(content=content, role=role)
