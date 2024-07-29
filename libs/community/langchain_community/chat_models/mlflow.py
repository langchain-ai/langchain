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
    cast,
)
from urllib.parse import urlparse

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
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
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
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)


class ChatMlflow(BaseChatModel):
    """`MLflow` chat models API.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatMlflow

            chat = ChatMlflow(
                target_uri="http://localhost:5000",
                endpoint="chat",
                temperature-0.1,
            )
    """

    endpoint: str
    """The endpoint to use."""
    target_uri: str
    """The target URI to use."""
    temperature: float = 0.0
    """The sampling temperature."""
    n: int = 1
    """The number of completion choices to generate."""
    stop: Optional[List[str]] = None
    """The stop sequence."""
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    extra_params: dict = Field(default_factory=dict)
    """Any extra parameters to pass to the endpoint."""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._validate_uri()
        try:
            from mlflow.deployments import get_deploy_client

            self._client = get_deploy_client(self.target_uri)
        except ImportError as e:
            raise ImportError(
                "Failed to create the client. "
                f"Please run `pip install mlflow{self._mlflow_extras}` to install "
                "required dependencies."
            ) from e

    @property
    def _mlflow_extras(self) -> str:
        return "[genai]"

    def _validate_uri(self) -> None:
        if self.target_uri == "databricks":
            return
        allowed = ["http", "https", "databricks"]
        if urlparse(self.target_uri).scheme not in allowed:
            raise ValueError(
                f"Invalid target URI: {self.target_uri}. "
                f"The scheme must be one of {allowed}."
            )

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

    def _prepare_inputs(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        message_dicts = [
            ChatMlflow._convert_message_to_dict(message) for message in messages
        ]
        data: Dict[str, Any] = {
            "messages": message_dicts,
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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        data = self._prepare_inputs(
            messages,
            stop,
            **kwargs,
        )
        resp = self._client.predict(endpoint=self.endpoint, inputs=data)
        return ChatMlflow._create_chat_result(resp)

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        # We need to override `stream` to handle the case
        # that `self._client` does not implement `predict_stream`
        if not hasattr(self._client, "predict_stream"):
            # MLflow deployment client does not implement streaming,
            # so use default implementation
            yield cast(
                BaseMessageChunk, self.invoke(input, config=config, stop=stop, **kwargs)
            )
        else:
            yield from super().stream(input, config, stop=stop, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        data = self._prepare_inputs(
            messages,
            stop,
            **kwargs,
        )
        # TODO: check if `_client.predict_stream` is available.
        chunk_iter = self._client.predict_stream(endpoint=self.endpoint, inputs=data)
        first_chunk_role = None
        for chunk in chunk_iter:
            if chunk["choices"]:
                choice = chunk["choices"][0]

                chunk_delta = choice["delta"]
                if first_chunk_role is None:
                    first_chunk_role = chunk_delta.get("role")

                chunk_message = ChatMlflow._convert_delta_to_message_chunk(
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

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self._default_params

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model FOR THE CALLBACKS."""
        return {
            **self._default_params,
            **super()._get_invocation_params(stop=stop, **kwargs),
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mlflow-chat"

    @staticmethod
    def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
        role = _dict["role"]
        content = cast(str, _dict.get("content"))
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            content = content or ""
            additional_kwargs: Dict = {}
            tool_calls = []
            invalid_tool_calls = []
            if raw_tool_calls := _dict.get("tool_calls"):
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    try:
                        tool_calls.append(
                            parse_tool_call(raw_tool_call, return_id=True)
                        )
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
        elif role == "system":
            return SystemMessage(content=content)
        else:
            return ChatMessage(content=content, role=role)

    @staticmethod
    def _convert_delta_to_message_chunk(
        _dict: Mapping[str, Any], default_role: str
    ) -> BaseMessageChunk:
        role = _dict.get("role", default_role)
        content = _dict.get("content") or ""
        if role == "user":
            return HumanMessageChunk(content=content)
        elif role == "assistant":
            additional_kwargs: Dict = {}
            tool_call_chunks = []
            if raw_tool_calls := _dict.get("tool_calls"):
                additional_kwargs["tool_calls"] = raw_tool_calls
                try:
                    tool_call_chunks = [
                        tool_call_chunk(
                            name=rtc["function"].get("name"),
                            args=rtc["function"].get("arguments"),
                            id=rtc.get("id"),
                            index=rtc["index"],
                        )
                        for rtc in raw_tool_calls
                    ]
                except KeyError:
                    pass
            return AIMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                id=_dict.get("id"),
                tool_call_chunks=tool_call_chunks,
            )
        elif role == "system":
            return SystemMessageChunk(content=content)
        elif role == "tool":
            return ToolMessageChunk(
                content=content, tool_call_id=_dict["tool_call_id"], id=_dict.get("id")
            )
        else:
            return ChatMessageChunk(content=content, role=role)

    @staticmethod
    def _raise_functions_not_supported() -> None:
        raise ValueError(
            "Function messages are not supported by Databricks. Please"
            " create a feature request at https://github.com/mlflow/mlflow/issues."
        )

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> dict:
        message_dict = {"content": message.content}
        if (name := message.name or message.additional_kwargs.get("name")) is not None:
            message_dict["name"] = name
        if isinstance(message, ChatMessage):
            message_dict["role"] = message.role
        elif isinstance(message, HumanMessage):
            message_dict["role"] = "user"
        elif isinstance(message, AIMessage):
            message_dict["role"] = "assistant"
            if message.tool_calls or message.invalid_tool_calls:
                message_dict["tool_calls"] = [
                    _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
                ] + [
                    _lc_invalid_tool_call_to_openai_tool_call(tc)
                    for tc in message.invalid_tool_calls
                ]  # type: ignore[assignment]
            elif "tool_calls" in message.additional_kwargs:
                message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
                tool_call_supported_props = {"id", "type", "function"}
                message_dict["tool_calls"] = [
                    {
                        k: v
                        for k, v in tool_call.items()  # type: ignore[union-attr]
                        if k in tool_call_supported_props
                    }
                    for tool_call in message_dict["tool_calls"]
                ]
            else:
                pass
            # If tool calls present, content null value should be None not empty string.
            if "tool_calls" in message_dict:
                message_dict["content"] = message_dict["content"] or None  # type: ignore[assignment]
        elif isinstance(message, SystemMessage):
            message_dict["role"] = "system"
        elif isinstance(message, ToolMessage):
            message_dict["role"] = "tool"
            message_dict["tool_call_id"] = message.tool_call_id
            supported_props = {"content", "role", "tool_call_id"}
            message_dict = {
                k: v for k, v in message_dict.items() if k in supported_props
            }
        elif isinstance(message, FunctionMessage):
            raise ValueError(
                "Function messages are not supported by Databricks. Please"
                " create a feature request at https://github.com/mlflow/mlflow/issues."
            )
        else:
            raise ValueError(f"Got unknown message type: {message}")

        if "function_call" in message.additional_kwargs:
            ChatMlflow._raise_functions_not_supported()
        return message_dict

    @staticmethod
    def _create_chat_result(response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for choice in response["choices"]:
            message = ChatMlflow._convert_dict_to_message(choice["message"])
            usage = choice.get("usage", {})
            gen = ChatGeneration(
                message=message,
                generation_info=usage,
            )
            generations.append(gen)

        usage = response.get("usage", {})
        return ChatResult(generations=generations, llm_output=usage)

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


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
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
