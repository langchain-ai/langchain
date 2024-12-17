import json
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
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
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_REKA_MODEL = "reka-flash"

ContentType = Union[str, List[Union[str, Dict[str, Any]]]]


def process_content_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single content item."""
    if item["type"] == "image_url":
        image_url = item["image_url"]
        if isinstance(image_url, dict) and "url" in image_url:
            # If it's in LangChain format, extract the URL value
            item["image_url"] = image_url["url"]
    return item


def process_content(content: ContentType) -> List[Dict[str, Any]]:
    """Process content to handle both text and media inputs,
    returning a list of content items."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    elif isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, str):
                result.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                result.append(process_content_item(item))
            else:
                raise ValueError(f"Invalid content item format: {item}")
        return result
    else:
        raise ValueError("Invalid content format")


def convert_to_reka_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert LangChain messages to Reka message format."""
    reka_messages: List[Dict[str, Any]] = []
    system_message: Optional[str] = None

    for message in messages:
        if isinstance(message, SystemMessage):
            if system_message is None:
                if isinstance(message.content, str):
                    system_message = message.content
                else:
                    raise TypeError("SystemMessage content must be a string.")
            else:
                raise ValueError("Multiple system messages are not supported.")
        elif isinstance(message, HumanMessage):
            processed_content = process_content(message.content)
            if system_message:
                if (
                    processed_content
                    and isinstance(processed_content[0], dict)
                    and processed_content[0].get("type") == "text"
                    and "text" in processed_content[0]
                ):
                    processed_content[0]["text"] = (
                        f"{system_message}\n{processed_content[0]['text']}"
                    )
                else:
                    processed_content.insert(
                        0, {"type": "text", "text": system_message}
                    )
                system_message = None
            reka_messages.append({"role": "user", "content": processed_content})
        elif isinstance(message, AIMessage):
            reka_message: Dict[str, Any] = {"role": "assistant"}
            if message.content:
                processed_content = process_content(message.content)
                reka_message["content"] = processed_content
            if "tool_calls" in message.additional_kwargs:
                tool_calls = message.additional_kwargs["tool_calls"]
                formatted_tool_calls = []
                for tool_call in tool_calls:
                    formatted_tool_call = {
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "parameters": json.loads(tool_call["function"]["arguments"]),
                    }
                    formatted_tool_calls.append(formatted_tool_call)
                reka_message["tool_calls"] = formatted_tool_calls
            reka_messages.append(reka_message)
        elif isinstance(message, ToolMessage):
            content_list: List[Dict[str, Any]] = []
            content_list.append(
                {
                    "tool_call_id": message.tool_call_id,
                    "output": json.dumps({"status": message.content}),
                }
            )
            reka_messages.append(
                {
                    "role": "tool_output",
                    "content": content_list,
                }
            )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    return reka_messages


class ChatReka(BaseChatModel):
    """Reka chat large language models."""

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: str = Field(default=DEFAULT_REKA_MODEL)
    max_tokens: int = Field(default=256)
    temperature: Optional[float] = None
    streaming: bool = False
    default_request_timeout: Optional[float] = None
    max_retries: int = 2
    reka_api_key: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")
    token_counter: Optional[
        Callable[[Union[str, BaseMessage, List[BaseMessage]]], int]
    ] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that API key and Python package exist in the environment."""
        reka_api_key = values.get("reka_api_key")
        reka_api_key = get_from_dict_or_env(
            {"reka_api_key": reka_api_key}, "reka_api_key", "REKA_API_KEY"
        )
        values["reka_api_key"] = reka_api_key

        try:
            # Import reka libraries here
            from reka.client import AsyncReka, Reka

            values["client"] = Reka(
                api_key=reka_api_key,
            )
            values["async_client"] = AsyncReka(
                api_key=reka_api_key,
            )
        except ImportError:
            raise ImportError(
                "Could not import Reka Python package. "
                "Please install it with `pip install reka-api`."
            )
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Reka API."""
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return {**params, **self.model_kwargs}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "reka-chat"

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
            chat_chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=chat_chunk)
            yield chat_chunk

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
            chat_chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            if run_manager:
                await run_manager.on_llm_new_token(content, chunk=chat_chunk)
            yield chat_chunk

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

    def get_num_tokens(self, input: Union[str, BaseMessage, List[BaseMessage]]) -> int:
        """Calculate number of tokens.

        Args:
            input: Either a string, a single BaseMessage, or a list of BaseMessages.

        Returns:
            int: Number of tokens in the input.

        Raises:
            ImportError: If tiktoken is not installed.
            ValueError: If message content is not a string.
        """
        if self.token_counter is not None:
            return self.token_counter(input)

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "Please install it with `pip install tiktoken`."
            )

        encoding = tiktoken.get_encoding("cl100k_base")

        if isinstance(input, str):
            return len(encoding.encode(input))
        elif isinstance(input, BaseMessage):
            content = input.content
            if not isinstance(content, str):
                raise ValueError(
                    f"Message content must be a string, got {type(content)}"
                )
            return len(encoding.encode(content))
        elif isinstance(input, list):
            total = 0
            for msg in input:
                content = msg.content
                if not isinstance(content, str):
                    raise ValueError(
                        f"Message content must be a string, got {type(content)}"
                    )
                total += len(encoding.encode(content))
            return total
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: str = "auto",
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        The `tool_choice` parameter controls how the model uses the tools you pass.
        There are three available options:

        - `"auto"`: Lets the model decide whether or not to invoke a tool. This is the
          recommended way to do function calling with our models.
        - `"none"`: Disables tool calling. In this case, even if you pass tools to
          the model, the model will not invoke any tools.
        - `"tool"`: Forces the model to invoke one or more of the tools it has
          been passed.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Controls how the model uses the tools you pass.
                Options are "auto", "none", or "tool". Defaults to "auto".
            strict:
            If True, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition.
                If False, input schema will not be validated
                and model output will not be validated.
                If None, ``strict`` argument will not
                be passed to the model.
            kwargs: Any additional parameters are passed directly to the model.

        Returns:
            Runnable: An executable chain or component.
        """
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]

        # Ensure tool_choice is one of the allowed options
        if tool_choice not in ("auto", "none", "tool"):
            raise ValueError(
                f"Invalid tool_choice '{tool_choice}' provided. "
                "Tool choice must be one of: 'auto', 'none', or 'tool'."
            )

        # Map tool_choice to the parameter expected by the Reka API
        kwargs["tool_choice"] = tool_choice

        # Pass the tools and updated kwargs to the model
        formatted_tools = [tool["function"] for tool in formatted_tools]
        return super().bind(tools=formatted_tools, **kwargs)
