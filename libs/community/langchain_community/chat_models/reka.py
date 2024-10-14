import json
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
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
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
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.utils import build_extra_kwargs, convert_to_secret_str
from pydantic import Field, SecretStr, model_validator

try:
    from reka import ChatMessage, ToolCall
    from reka.client import AsyncReka, Reka
except ImportError:
    raise ValueError(
        "Reka is not installed. Please install it with `pip install reka-api`."
    )

REKA_MODELS = [
    "reka-edge",
    "reka-flash",
    "reka-core",
]

DEFAULT_REKA_MODEL = "reka-flash"


def process_content_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single content item."""
    if item["type"] == "image_url":
        image_url = item["image_url"]
        if isinstance(image_url, dict) and "url" in image_url:
            # If it's in LangChain format, extract the URL value
            item["image_url"] = image_url["url"]
    return item


def process_content(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Process content to handle both text and media inputs,
    Returning a list of content items."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    elif isinstance(content, list):
        return [process_content_item(item) for item in content]
    else:
        raise ValueError("Invalid content format")


def convert_to_reka_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert LangChain messages to Reka message format."""
    reka_messages = []
    system_message = None  # Double check on the system message

    for message in messages:
        if isinstance(message, SystemMessage):
            if system_message is None:
                system_message = message.content
            else:
                raise ValueError("Multiple system messages are not supported.")
        elif isinstance(message, HumanMessage):
            content = process_content(message.content)
            if system_message:
                if isinstance(content[0], dict) and content[0].get("type") == "text":
                    content[0]["text"] = f"{system_message}\n{content[0]['text']}"
                else:
                    content.insert(0, {"type": "text", "text": system_message})
                system_message = None
            reka_messages.append({"role": "user", "content": content})
        elif isinstance(message, AIMessage):
            reka_message = {"role": "assistant"}
            if message.content:
                reka_message["content"] = process_content(message.content)

            if "tool_calls" in message.additional_kwargs:
                tool_calls = message.additional_kwargs["tool_calls"]
                formatted_tool_calls = []
                for tool_call in tool_calls:
                    formatted_tool_call = ToolCall(
                        id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        parameters=json.loads(tool_call["function"]["arguments"]),
                    )
                    formatted_tool_calls.append(formatted_tool_call)
                reka_message["tool_calls"] = formatted_tool_calls
            reka_messages.append(reka_message)
        elif isinstance(message, ToolMessage):
            reka_messages.append(
                {
                    "role": "tool_output",
                    "content": [
                        {
                            "tool_call_id": message.tool_call_id,
                            "output": json.dumps({"status": message.content}),
                        }
                    ],
                }
            )
        elif isinstance(message, ChatMessage):
            content = process_content(message.content)
            reka_messages.append({"role": message.role, "content": content})
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    return reka_messages


class RekaCommon(BaseLanguageModel):
    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: str = Field(default=DEFAULT_REKA_MODEL, alias="model_name")
    max_tokens: int = Field(default=256)
    temperature: Optional[float] = None
    streaming: bool = False
    default_request_timeout: Optional[float] = None
    max_retries: int = 2
    reka_api_key: Optional[SecretStr] = None
    count_tokens: Optional[Callable[[str], int]] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    def build_extra(cls, values: Dict) -> Dict:
        extra = values.get("model_kwargs", {})
        all_required_field_names = get_pydantic_field_names(cls)
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @model_validator(mode="after")
    def validate_environment(cls, self: "RekaCommon") -> "RekaCommon":
        """Validate that API key and Python package exist in the environment."""
        self.reka_api_key = convert_to_secret_str(
            get_from_dict_or_env(self, "reka_api_key", "REKA_API_KEY")
        )

        try:
            self.client = Reka(
                api_key=self.reka_api_key.get_secret_value(),
            )
            self.async_client = AsyncReka(
                api_key=self.reka_api_key.get_secret_value(),
            )

        except ImportError:
            raise ImportError(
                "Could not import Reka Python package. "
                "Please install it with `pip install reka-api`."
            )
        return self

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Reka API."""
        d = {
            "max_tokens": self.max_tokens,
            "model": self.model,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        return {**d, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{}, **self._default_params}


class ChatReka(BaseChatModel, RekaCommon):
    """Reka chat large language models."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "reka-chat"

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Reka API."""
        d = {
            "max_tokens": self.max_tokens,
            "model": self.model,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature

        return {**d, **self.model_kwargs}

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reka_messages = convert_to_reka_messages(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop

        stream = self.client.chat.create_stream(messages=reka_messages, **params)

        for chunk in stream:
            content = chunk.responses[0].chunk.content
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        reka_messages = convert_to_reka_messages(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop

        stream = self.async_client.chat.create_stream(messages=reka_messages, **params)

        async for chunk in stream:
            content = chunk.responses[0].chunk.content
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(content, chunk=chunk)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return generate_from_stream(
                self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )

        reka_messages = convert_to_reka_messages(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop
        response = self.client.chat.create(messages=reka_messages, **params)

        if response.responses[0].message.tool_calls:
            tool_calls = response.responses[0].message.tool_calls
            message = AIMessage(
                content="",  # Empty string instead of None
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.parameters),
                            },
                        }
                        for tc in tool_calls
                    ]
                },
            )
        else:
            content = response.responses[0].message.content
            # Ensure content is never None
            message = AIMessage(content=content if content is not None else "")

        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return await agenerate_from_stream(
                self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )

        reka_messages = convert_to_reka_messages(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop
        response = await self.async_client.chat.create(messages=reka_messages, **params)

        if response.responses[0].message.tool_calls:
            tool_calls = response.responses[0].message.tool_calls
            message = AIMessage(
                content="",  # Empty string instead of None
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.parameters),
                            },
                        }
                        for tc in tool_calls
                    ]
                },
            )
        else:
            content = response.responses[0].message.content
            # Ensure content is never None
            message = AIMessage(content=content if content is not None else "")

        return ChatResult(generations=[ChatGeneration(message=message)])

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        if self.count_tokens is None:
            raise NotImplementedError(
                "get_num_tokens() is not implemented for Reka models."
            )
        return self.count_tokens(text)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = "auto",
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call. Options are:

                - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                - ``"auto"``: automatically selects a tool (including no tool).
                - ``"none"``: does not call a tool.
                - ``"any"`` or ``"required"`` or ``True``: force at least one tool to be called.
                - dict of the form ``{"type": "function", "function": {"name": <<tool_name>>}}``: calls <<tool_name>> tool.
                - ``False`` or ``None``: no effect, default OpenAI behavior.
            strict: If True, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition. If True, the input schema will be
                validated according to
                https://platform.openai.com/docs/guides/structured-outputs/supported-schemas.
                If False, input schema will not be validated and model output will not
                be validated.
                If None, ``strict`` argument will not be passed to the model.
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_openai.chat_models.base.ChatOpenAI.bind`.

        .. versionchanged:: 0.1.21

            Support for ``strict`` argument added.

        """  # noqa: E501

        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
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
        # Formatting hack TODO
        formatted_tools = [
            formatted_tool["function"] for formatted_tool in formatted_tools
        ]
        return super().bind(tools=formatted_tools, **kwargs)
