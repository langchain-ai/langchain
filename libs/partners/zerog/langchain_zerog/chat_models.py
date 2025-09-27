"""0G Compute Network chat models."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, Dict, List, Optional, Union, Literal, TypeVar

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LangSmithParams, LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
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
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_zerog.broker import ZeroGBroker, OFFICIAL_MODELS

logger = logging.getLogger(__name__)

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[dict[str, Any], type[_BM], type]
_DictOrPydantic = Union[dict, _BM]


class ChatZeroG(BaseChatModel):
    """0G Compute Network chat model integration.

    The 0G Compute Network provides decentralized AI inference services with
    verified computations running in Trusted Execution Environments (TEE).

    Setup:
        Install ``langchain-zerog`` and set environment variables.

        .. code-block:: bash

            pip install -U langchain-zerog
            export ZEROG_PRIVATE_KEY="your-ethereum-private-key"
            export ZEROG_RPC_URL="https://evmrpc-testnet.0g.ai"  # Optional

    Key init args — completion params:
        model: str
            Name of 0G model to use. Supported models:
            - "llama-3.3-70b-instruct": 70B parameter model for general AI tasks
            - "deepseek-r1-70b": Advanced reasoning model
        provider_address: Optional[str]
            Specific provider address to use. If not provided, will use the
            official provider for the specified model.
        temperature: float
            Sampling temperature between 0 and 2.
        max_tokens: Optional[int]
            Maximum number of tokens to generate.
        top_p: float
            Nucleus sampling parameter.
        frequency_penalty: float
            Frequency penalty parameter.
        presence_penalty: float
            Presence penalty parameter.

    Key init args — client params:
        private_key: Optional[str]
            Ethereum private key for wallet authentication. If not provided,
            will read from ZEROG_PRIVATE_KEY environment variable.
        rpc_url: str
            0G Network RPC URL. Defaults to testnet.
        broker_url: str
            0G broker service URL.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_zerog import ChatZeroG

            llm = ChatZeroG(
                model="llama-3.3-70b-instruct",
                temperature=0.7,
                max_tokens=1000,
                # private_key="...",  # Or set ZEROG_PRIVATE_KEY
            )

    Fund Account (Required):
        .. code-block:: python

            # Fund your account before first use
            await llm.fund_account("0.1")  # Add 0.1 OG tokens

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful AI assistant."),
                ("human", "What is the capital of France?"),
            ]
            response = await llm.ainvoke(messages)
            print(response.content)

    Stream:
        .. code-block:: python

            async for chunk in llm.astream(messages):
                print(chunk.content, end="")

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''
                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            llm_with_tools = llm.bind_tools([GetWeather])
            ai_msg = await llm_with_tools.ainvoke("What's the weather in SF?")
            print(ai_msg.tool_calls)

    Structured output:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            structured_llm = llm.with_structured_output(Joke)
            result = await structured_llm.ainvoke("Tell me a joke about cats")

    Account Management:
        .. code-block:: python

            # Check balance
            balance = await llm.get_balance()
            print(f"Available: {balance['available']} OG")

            # Request refund
            await llm.request_refund("0.1")

    Token usage:
        .. code-block:: python

            response = await llm.ainvoke(messages)
            print(response.usage_metadata)

    Response metadata:
        .. code-block:: python

            response = await llm.ainvoke(messages)
            print(response.response_metadata)
    """

    model: str = Field(description="Name of 0G model to use")
    """Name of the 0G model to use."""

    provider_address: Optional[str] = Field(
        default=None,
        description="Specific provider address to use"
    )
    """Provider address. If not provided, uses official provider for the model."""

    private_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("ZEROG_PRIVATE_KEY", default=None),
        description="Ethereum private key for wallet authentication"
    )
    """Ethereum private key for wallet authentication."""

    rpc_url: str = Field(
        default_factory=from_env("ZEROG_RPC_URL", default="https://evmrpc-testnet.0g.ai"),
        description="0G Network RPC URL"
    )
    """0G Network RPC URL."""

    broker_url: str = Field(
        default_factory=from_env("ZEROG_BROKER_URL", default="https://broker.0g.ai"),
        description="0G broker service URL"
    )
    """0G broker service URL."""

    temperature: float = Field(default=0.7, description="Sampling temperature")
    """Sampling temperature between 0 and 2."""

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    """Maximum number of tokens to generate."""

    top_p: float = Field(default=1.0, description="Nucleus sampling parameter")
    """Nucleus sampling parameter."""

    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    """Frequency penalty parameter."""

    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    """Presence penalty parameter."""

    streaming: bool = Field(default=False, description="Whether to stream responses")
    """Whether to stream responses."""

    n: int = Field(default=1, description="Number of completions to generate")
    """Number of completions to generate."""

    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    """Stop sequences."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ChatZeroG."""
        super().__init__(**kwargs)
        self._broker: Optional[ZeroGBroker] = None

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-zerog"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"private_key": "ZEROG_PRIVATE_KEY"}

    def _get_ls_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get LangSmith parameters."""
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "zerog"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment and initialize broker."""
        if not (self.private_key and self.private_key.get_secret_value()):
            msg = "ZEROG_PRIVATE_KEY must be set."
            raise ValueError(msg)

        # Set provider address if not provided
        if not self.provider_address and self.model in OFFICIAL_MODELS:
            self.provider_address = OFFICIAL_MODELS[self.model]["provider_address"]
        elif not self.provider_address:
            msg = f"Provider address not found for model {self.model}. Please specify provider_address."
            raise ValueError(msg)

        return self

    def _get_broker(self) -> ZeroGBroker:
        """Get or create the 0G broker instance."""
        if self._broker is None:
            if not self.private_key:
                msg = "Private key is required"
                raise ValueError(msg)
            self._broker = ZeroGBroker(
                private_key=self.private_key.get_secret_value(),
                rpc_url=self.rpc_url,
                broker_url=self.broker_url,
            )
        return self._broker

    async def fund_account(self, amount: str) -> Dict[str, Any]:
        """Add funds to the account.

        Args:
            amount: Amount of OG tokens to add (e.g., "0.1")

        Returns:
            Transaction result
        """
        broker = self._get_broker()
        return await broker.fund_account(amount)

    async def get_balance(self) -> Dict[str, str]:
        """Get account balance information.

        Returns:
            Dictionary with balance, locked, and available amounts
        """
        broker = self._get_broker()
        return await broker.get_balance()

    async def request_refund(self, amount: str) -> Dict[str, Any]:
        """Request a refund for unused funds.

        Args:
            amount: Amount to refund

        Returns:
            Refund result
        """
        broker = self._get_broker()
        return await broker.request_refund("inference", amount)

    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI format."""
        openai_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                openai_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                openai_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                msg_dict = {"role": "assistant", "content": message.content}
                if message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in message.tool_calls
                    ]
                openai_messages.append(msg_dict)
            elif isinstance(message, ToolMessage):
                openai_messages.append({
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                })
        return openai_messages

    def _create_chat_request(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        """Create the request payload for the 0G service."""
        openai_messages = self._convert_messages_to_openai_format(messages)

        request_data = {
            "messages": openai_messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "stream": self.streaming,
        }

        # Add stop sequences if provided
        if self.stop:
            request_data["stop"] = self.stop

        # Add any additional kwargs
        request_data.update(kwargs)

        # Remove None values
        return {k: v for k, v in request_data.items() if v is not None}

    async def _make_request(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a request to the 0G Compute Network."""
        broker = self._get_broker()

        # Ensure provider is acknowledged
        if self.provider_address:
            await broker.acknowledge_provider(self.provider_address)

        # Get service metadata
        metadata = await broker.get_service_metadata(self.provider_address or "")
        endpoint = metadata["endpoint"]

        # Create request payload
        request_payload = self._create_chat_request(messages, **kwargs)

        # Get authenticated headers
        content = json.dumps(request_payload)
        headers = await broker.get_request_headers(
            self.provider_address or "",
            content
        )
        headers["Content-Type"] = "application/json"

        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/chat/completions",
                headers=headers,
                data=content,
            ) as response:
                response.raise_for_status()
                response_data = await response.json()

        # Process response for verification
        await broker.process_response(
            self.provider_address or "",
            json.dumps(response_data),
        )

        return response_data

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a response using the 0G Compute Network."""
        # Override stop sequences if provided
        if stop:
            kwargs["stop"] = stop

        response_data = await self._make_request(messages, **kwargs)

        generations = []
        for choice in response_data["choices"]:
            message_content = choice["message"]["content"]

            # Create AI message
            ai_message = AIMessage(content=message_content)

            # Add tool calls if present
            if "tool_calls" in choice["message"]:
                tool_calls = []
                for tc in choice["message"]["tool_calls"]:
                    tool_calls.append({
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]),
                    })
                ai_message.tool_calls = tool_calls

            # Extract usage information if available
            usage = response_data.get("usage", {})
            generation_info = {
                "finish_reason": choice.get("finish_reason"),
                "model": response_data.get("model", self.model),
            }

            if usage:
                ai_message.usage_metadata = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            ai_message.response_metadata = {
                "provider_address": self.provider_address,
                "model": self.model,
                "verification": OFFICIAL_MODELS.get(self.model, {}).get("verification", "None"),
                **generation_info,
            }

            generations.append(ChatGeneration(message=ai_message, generation_info=generation_info))

        return ChatResult(generations=generations)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the 0G Compute Network."""
        import asyncio

        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we can't use run_until_complete
            msg = (
                "Cannot call synchronous _generate from within an async context. "
                "Use ainvoke or agenerate instead."
            )
            raise RuntimeError(msg)
        except RuntimeError:
            # No running loop, we can create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    msg = (
                        "Cannot call synchronous _generate from within an async context. "
                        "Use ainvoke or agenerate instead."
                    )
                    raise RuntimeError(msg)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream responses from the 0G Compute Network."""
        # Set streaming to True for this request
        kwargs["stream"] = True

        broker = self._get_broker()

        # Ensure provider is acknowledged
        if self.provider_address:
            await broker.acknowledge_provider(self.provider_address)

        # Get service metadata
        metadata = await broker.get_service_metadata(self.provider_address or "")
        endpoint = metadata["endpoint"]

        # Create request payload
        request_payload = self._create_chat_request(messages, **kwargs)

        # Get authenticated headers
        content = json.dumps(request_payload)
        headers = await broker.get_request_headers(
            self.provider_address or "",
            content
        )
        headers["Content-Type"] = "application/json"

        # Make streaming request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/chat/completions",
                headers=headers,
                data=content,
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:
                                        chunk = AIMessageChunk(content=content)
                                        yield ChatGenerationChunk(message=chunk)
                        except json.JSONDecodeError:
                            continue

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from the 0G Compute Network."""
        import asyncio

        async def _async_generator():
            async for chunk in self._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk

        # Convert async generator to sync
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                msg = (
                    "Cannot call synchronous _stream from within an async context. "
                    "Use astream instead."
                )
                raise RuntimeError(msg)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async_gen = _async_generator()
        try:
            while True:
                yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            pass

    def bind_tools(
        self,
        tools: List[Union[Dict[str, Any], type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool schemas to the model."""
        # Convert tools to OpenAI format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                # Pydantic model
                schema = tool.model_json_schema()
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": tool.__doc__ or "",
                        "parameters": schema,
                    },
                })
            elif isinstance(tool, BaseTool):
                # LangChain tool
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {},
                    },
                })

        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema."""
        if method == "function_calling":
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                tool_name = schema.__name__
                tool_description = schema.__doc__ or ""

                llm = self.bind_tools([schema], **kwargs)

                if include_raw:
                    def _parse_with_raw(ai_message: BaseMessage) -> Dict[str, Any]:
                        try:
                            if ai_message.tool_calls:
                                return {
                                    "raw": ai_message,
                                    "parsed": schema(**ai_message.tool_calls[0]["args"]),
                                    "parsing_error": None,
                                }
                            else:
                                return {
                                    "raw": ai_message,
                                    "parsed": None,
                                    "parsing_error": ValueError("No tool calls found"),
                                }
                        except Exception as e:
                            return {
                                "raw": ai_message,
                                "parsed": None,
                                "parsing_error": e,
                            }

                    return llm | _parse_with_raw
                else:
                    def _parse(ai_message: BaseMessage) -> schema:
                        if ai_message.tool_calls:
                            return schema(**ai_message.tool_calls[0]["args"])
                        else:
                            msg = "No tool calls found in response"
                            raise ValueError(msg)

                    return llm | _parse

        msg = f"Method {method} not supported"
        raise NotImplementedError(msg)
