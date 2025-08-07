"""v1 Ollama implementation.

Provides native support for v1 messages with standard content blocks.

.. versionadded:: 1.0.0

"""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from operator import itemgetter
from typing import Any, Callable, Literal, Optional, Union, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from langchain_core.v1.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.v1.messages import AIMessage, AIMessageChunk, MessageV1
from ollama import AsyncClient, Client, Options
from pydantic import BaseModel, PrivateAttr, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self, is_typeddict

from langchain_ollama._compat import (
    _convert_chunk_to_v1,
    _convert_from_v1_to_ollama_format,
    _convert_to_v1_from_ollama_format,
)
from langchain_ollama._utils import validate_model

log = logging.getLogger(__name__)


def _parse_json_string(
    json_string: str,
    *,
    raw_tool_call: dict[str, Any],
    skip: bool,
) -> Any:
    """Attempt to parse a JSON string for tool calling.

    It first tries to use the standard ``json.loads``. If that fails, it falls
    back to ``ast.literal_eval`` to safely parse Python literals, which is more
    robust against models using single quotes or containing apostrophes.

    Args:
        json_string: JSON string to parse.
        raw_tool_call: Raw tool call to include in error message.
        skip: Whether to ignore parsing errors and return the value anyways.

    Returns:
        The parsed JSON string or Python literal.

    Raises:
        OutputParserException: If the string is invalid and ``skip=False``.

    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
            # Use ast.literal_eval to safely parse Python-style dicts
            # (e.g. with single quotes)
            return ast.literal_eval(json_string)
        except (SyntaxError, ValueError) as e:
            # If both fail, and we're not skipping, raise an informative error.
            if skip:
                return json_string
            msg = (
                f"Function {raw_tool_call['function']['name']} arguments:\n\n"
                f"{raw_tool_call['function']['arguments']}"
                "\n\nare not valid JSON or a Python literal. "
                f"Received error: {e}"
            )
            raise OutputParserException(msg) from e
    except TypeError as e:
        if skip:
            return json_string
        msg = (
            f"Function {raw_tool_call['function']['name']} arguments:\n\n"
            f"{raw_tool_call['function']['arguments']}\n\nare not a string or a "
            f"dictionary. Received TypeError {e}"
        )
        raise OutputParserException(msg) from e


def _parse_arguments_from_tool_call(
    raw_tool_call: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Parse arguments by trying to parse any shallowly nested string-encoded JSON.

    Band-aid fix for issue in Ollama with inconsistent tool call argument structure.
    Should be removed/changed if fixed upstream.

    `See #6155 <https://github.com/ollama/ollama/issues/6155>`__.

    """
    if "function" not in raw_tool_call:
        return None
    arguments = raw_tool_call["function"]["arguments"]
    parsed_arguments: dict = {}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if isinstance(value, str):
                parsed_value = _parse_json_string(
                    value, skip=True, raw_tool_call=raw_tool_call
                )
                if isinstance(parsed_value, (dict, list)):
                    parsed_arguments[key] = parsed_value
                else:
                    parsed_arguments[key] = value
            else:
                parsed_arguments[key] = value
    else:
        parsed_arguments = _parse_json_string(
            arguments, skip=False, raw_tool_call=raw_tool_call
        )
    return parsed_arguments


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class ChatOllama(BaseChatModel):
    r"""Ollama chat model with v1 message/content block support.

    This implementation provides native support for structured content blocks.

    .. dropdown:: Setup
        :open:

        Install ``langchain-ollama`` and download any models you want to use from ollama.

        .. code-block:: bash

            ollama pull mistral:v0.3
            pip install -U langchain-ollama

    Key init args — completion params:
        model: str
            Name of Ollama model to use.
        reasoning: Optional[bool]
            Controls the reasoning/thinking mode for
            `supported models <https://ollama.com/search?c=thinking>`__.

            - ``True``: Enables reasoning mode. The model's reasoning process will be
              captured and returned as a ``ReasoningContentBlock`` in the response
              message content. The main response content will not include the reasoning tags.
            - ``False``: Disables reasoning mode. The model will not perform any reasoning,
              and the response will not include any reasoning content.
            - ``None`` (Default): The model will use its default reasoning behavior. Note
              however, if the model's default behavior *is* to perform reasoning, think tags
              (``<think>`` and ``</think>``) will be present within the main response ``TextContentBlock``s
              unless you set ``reasoning`` to ``True``.
        temperature: float
            Sampling temperature. Ranges from ``0.0`` to ``1.0``.
        num_predict: Optional[int]
            Max number of tokens to generate.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_ollama.v1 import ChatOllama

            llm = ChatOllama(
                model = "llama3",
                temperature = 0.8,
                num_predict = 256,
                # other params ...
            )

    Invoke:
        .. code-block:: python

            from langchain_core.v1.messages import HumanMessage
            from langchain_core.messages.content_blocks import TextContentBlock

            messages = [
                HumanMessage("Hello!")
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content=[{'type': 'text', 'text': 'Hello! How can I help you today?'}], ...)

    Stream:
        .. code-block:: python

            from langchain_core.v1.messages import HumanMessage
            from langchain_core.messages.content_blocks import TextContentBlock

            messages = [
                HumanMessage(Return the words Hello World!")
            ]
            for chunk in llm.stream(messages):
                print(chunk.content, end="")

        .. code-block:: python

            AIMessageChunk(content=[{'type': 'text', 'text': 'Hello'}], ...)
            AIMessageChunk(content=[{'type': 'text', 'text': ' World'}], ...)
            AIMessageChunk(content=[{'type': 'text', 'text': '!'}], ...)

    Multi-modal input:
        .. code-block:: python

            from langchain_core.messages.content_blocks import ImageContentBlock

            response = llm.invoke([
                HumanMessage(content=[
                    TextContentBlock(type="text", text="Describe this image:"),
                    ImageContentBlock(
                        type="image",
                        base64="base64_encoded_image",
                    )
                ])
            ])

    Tool Calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class Multiply(BaseModel):
                a: int = Field(..., description="First integer")
                b: int = Field(..., description="Second integer")

            llm_with_tools = llm.bind_tools([Multiply])
            ans = llm_with_tools.invoke([
                HumanMessage("What is 45*67")
            ])
            ans.tool_calls

        .. code-block:: python

            [
                {
                    'name': 'Multiply',
                    'args': {'a': 45, 'b': 67},
                    'id': '420c3f3b-df10-4188-945f-eb3abdb40622',
                    'type': 'tool_call'
                }
            ]

    """  # noqa: E501

    model: str
    """Model name to use."""

    streaming: bool = False
    """Whether to use streaming for invocation.

    If True, invoke will use streaming internally.

    """

    reasoning: Optional[bool] = None
    """Controls the reasoning/thinking mode for supported models.

    - ``True``: Enables reasoning mode. The model's reasoning process will be
      captured and returned as a ``ReasoningContentBlock`` in the response
      message content. The main response content will not include the reasoning tags.
    - ``False``: Disables reasoning mode. The model will not perform any reasoning,
      and the response will not include any reasoning content.
    - ``None`` (Default): The model will use its default reasoning behavior. Note
      however, if the model's default behavior *is* to perform reasoning, think tags
      (``<think>`` and ``</think>``) will be present within the main response content
      unless you set ``reasoning`` to ``True``.

    """

    validate_model_on_init: bool = False
    """Whether to validate the model exists in Ollama locally on initialization."""

    # Ollama-specific parameters
    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.

    (Default: ``0``, ``0`` = disabled, ``1`` = Mirostat, ``2`` = Mirostat 2.0)

    """

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback from generated text.

    A lower learning rate will result in slower adjustments, while a higher learning
    rate will make the algorithm more responsive.

    (Default: ``0.1``)

    """

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity of the output.

    A lower value will result in more focused and coherent text.

    (Default: ``5.0``)

    """

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the next token.

    (Default: ``2048``)

    """

    num_gpu: Optional[int] = None
    """The number of GPUs to use.

    On macOS it defaults to ``1`` to enable metal support, ``0`` to disable.

    """

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.

    By default, Ollama will detect this for optimal performance. It is recommended to
    set this value to the number of physical CPU cores your system has (as opposed to
    the logical number of cores).

    """

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict when generating text.

    (Default: ``128``, ``-1`` = infinite generation, ``-2`` = fill context)

    """

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent repetition.

    (Default: ``64``, ``0`` = disabled, ``-1`` = ``num_ctx``)

    """

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions.

    A higher value (e.g., ``1.5``) will penalize repetitions more strongly, while a
    lower value (e.g., ``0.9``) will be more lenient.

    (Default: ``1.1``)

    """

    temperature: Optional[float] = None
    """The temperature of the model.

    Increasing the temperature will make the model answer more creatively.

    (Default: ``0.8``)"""

    seed: Optional[int] = None
    """Sets the random number seed to use for generation.

    Setting this to a specific number will make the model generate the same text for the
    same prompt.

    """

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable tokens from the output.

    A higher value (e.g., ``2.0``) will reduce the impact more, while a value of ``1.0`` disables this setting.

    (Default: ``1``)

    """  # noqa: E501

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense.

    A higher value (e.g. ``100``) will give more diverse answers, while a lower value
    (e.g. ``10``) will be more conservative.

    (Default: ``40``)

    """

    top_p: Optional[float] = None
    """Works together with top-k.

    A higher value (e.g., ``0.95``) will lead to more diverse text, while a lower value
    (e.g., ``0.5``) will generate more focused and conservative text.

    (Default: ``0.9``)

    """

    format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None
    """Specify the format of the output (Options: ``'json'``, JSON schema)."""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx clients.

    These arguments are passed to both synchronous and async clients.

    Use ``sync_client_kwargs`` and ``async_client_kwargs`` to pass different arguments
    to synchronous and asynchronous clients.

    """

    async_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with ``client_kwargs`` before
    passing to the httpx AsyncClient.

    `Full list of params. <https://www.python-httpx.org/api/#asyncclient>`__

    """

    sync_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with ``client_kwargs`` before
    passing to the httpx Client.

    `Full list of params. <https://www.python-httpx.org/api/#client>`__

    """

    _client: Client = PrivateAttr()
    """The client to use for making requests."""

    _async_client: AsyncClient = PrivateAttr()
    """The async client to use for making requests."""

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients to use for ollama."""
        client_kwargs = self.client_kwargs or {}

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=self.base_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **async_client_kwargs)
        if self.validate_model_on_init:
            validate_model(self._client, self.model)
        return self

    def _get_ls_params(self, **kwargs: Any) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(**kwargs)
        ls_params = LangSmithParams(
            ls_provider="ollama",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_stop := params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _get_invocation_params(self, **kwargs: Any) -> dict[str, Any]:
        """Get parameters for model invocation."""
        params = {
            "model": self.model,
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "num_predict": self.num_predict,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "seed": self.seed,
            "tfs_z": self.tfs_z,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "format": self.format,
            "keep_alive": self.keep_alive,
        }
        params.update(kwargs)
        return params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ollama-v1"

    def _chat_params(
        self,
        messages: list[MessageV1],
        *,
        stream: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build parameters for Ollama chat API."""
        # Convert v1 messages to Ollama format
        ollama_messages = [_convert_from_v1_to_ollama_format(msg) for msg in messages]

        options_dict = kwargs.pop(
            "options",
            {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "seed": self.seed,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        )

        params = {
            "messages": ollama_messages,
            "stream": kwargs.pop("stream", stream),
            "model": kwargs.pop("model", self.model),
            "think": kwargs.pop("reasoning", self.reasoning),
            "format": kwargs.pop("format", self.format),
            "options": Options(**options_dict),
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

        if tools := kwargs.get("tools"):
            params["tools"] = tools

        return params

    def _generate_stream(
        self,
        messages: list[MessageV1],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """Generate streaming response with native v1 chunks."""
        chat_params = self._chat_params(messages, **kwargs)

        if chat_params["stream"]:
            for part in self._client.chat(**chat_params):
                if not isinstance(part, str):
                    # Skip empty load responses
                    if (
                        part.get("done") is True
                        and part.get("done_reason") == "load"
                        and not part.get("message", {}).get("content", "").strip()
                    ):
                        log.warning(
                            "Ollama returned empty response with `done_reason='load'`. "
                            "Skipping this response."
                        )
                        continue

                    chunk = _convert_chunk_to_v1(part)

                    if run_manager:
                        text_content = "".join(
                            str(block.get("text", ""))
                            for block in chunk.content
                            if block.get("type") == "text"
                        )
                        run_manager.on_llm_new_token(
                            text_content,
                            chunk=chunk,
                        )
                    yield chunk
        else:
            # Non-streaming case
            response = self._client.chat(**chat_params)
            ai_message = _convert_to_v1_from_ollama_format(response)
            chunk = AIMessageChunk(
                content=ai_message.content,
                response_metadata=ai_message.response_metadata,
                usage_metadata=ai_message.usage_metadata,
            )
            yield chunk

    async def _agenerate_stream(
        self,
        messages: list[MessageV1],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Generate async streaming response with native v1 chunks."""
        chat_params = self._chat_params(messages, **kwargs)

        if chat_params["stream"]:
            async for part in await self._async_client.chat(**chat_params):
                if not isinstance(part, str):
                    # Skip empty load responses
                    if (
                        part.get("done") is True
                        and part.get("done_reason") == "load"
                        and not part.get("message", {}).get("content", "").strip()
                    ):
                        log.warning(
                            "Ollama returned empty response with `done_reason='load'`. "
                            "Skipping this response."
                        )
                        continue

                    chunk = _convert_chunk_to_v1(part)

                    if run_manager:
                        text_content = "".join(
                            str(block.get("text", ""))
                            for block in chunk.content
                            if block.get("type") == "text"
                        )
                        await run_manager.on_llm_new_token(
                            text_content,
                            chunk=chunk,
                        )
                    yield chunk
        else:
            # Non-streaming case
            response = await self._async_client.chat(**chat_params)
            ai_message = _convert_to_v1_from_ollama_format(response)
            chunk = AIMessageChunk(
                content=ai_message.content,
                response_metadata=ai_message.response_metadata,
                usage_metadata=ai_message.usage_metadata,
            )
            yield chunk

    def _invoke(
        self,
        messages: list[MessageV1],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model with v1 messages and return a complete response.

        Args:
            messages: List of v1 format messages.
            run_manager: Callback manager for the run.
            kwargs: Additional parameters.

        Returns:
            Complete AI message response.

        """
        if self.streaming:
            stream_iter = self._stream(messages, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)

        chat_params = self._chat_params(messages, stream=False, **kwargs)
        response = self._client.chat(**chat_params)
        return _convert_to_v1_from_ollama_format(response)

    async def _ainvoke(
        self,
        messages: list[MessageV1],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke the model with v1 messages and return a complete response.

        Args:
            messages: List of v1 format messages.
            run_manager: Async callback manager for the run.
            kwargs: Additional parameters.

        Returns:
            Complete AI message response.

        """
        if self.streaming:
            stream_iter = self._astream(messages, run_manager=run_manager, **kwargs)
            return await agenerate_from_stream(stream_iter)

        # Non-streaming case: direct API call
        chat_params = self._chat_params(messages, stream=False, **kwargs)
        response = await self._async_client.chat(**chat_params)
        return _convert_to_v1_from_ollama_format(response)

    def _stream(
        self,
        messages: list[MessageV1],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """Stream response chunks using the v1 format.

        Args:
            messages: List of v1 format messages.
            run_manager: Callback manager for the run.
            kwargs: Additional parameters.

        Yields:
            AI message chunks in v1 format.

        """
        yield from self._generate_stream(messages, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: list[MessageV1],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Async stream response chunks using the v1 format.

        Args:
            messages: List of v1 format messages.
            run_manager: Async callback manager for the run.
            kwargs: Additional parameters.

        Yields:
            AI message chunks in v1 format.

        """
        async for chunk in self._agenerate_stream(
            messages, run_manager=run_manager, **kwargs
        ):
            yield chunk

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
            tool_choice: Tool choice parameter (currently ignored by Ollama).
            kwargs: Additional parameters passed to ``bind()``.

        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[dict, type],
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - a Pydantic class,
                - a JSON schema
                - a TypedDict class
                - an OpenAI function/tool schema.

                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method: The method for steering model generation, one of:

                - ``'json_schema'``:
                    Uses Ollama's `structured output API <https://ollama.com/blog/structured-outputs>`__
                - ``'function_calling'``:
                    Uses Ollama's tool-calling API
                - ``'json_mode'``:
                    Specifies ``format='json'``. Note that if using JSON mode then you
                    must include instructions for formatting the output into the
                    desired schema into the model call.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a ``BaseMessage``) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys ``'raw'``, ``'parsed'``, and ``'parsing_error'``.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs an instance of ``schema`` (i.e., a Pydantic object). Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:

            - ``'raw'``: ``BaseMessage``
            - ``'parsed'``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - ``'parsing_error'``: Optional[BaseException]

        .. versionchanged:: 0.2.2

            Added support for structured output API via ``format`` parameter.

        .. versionchanged:: 0.3.0

            Updated default ``method`` to ``'json_schema'``.

        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                format="json",
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            if is_pydantic_schema:
                schema = cast(TypeBaseModel, schema)
                if issubclass(schema, BaseModelV1):
                    response_format = schema.schema()
                else:
                    response_format = schema.model_json_schema()
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                )
                output_parser = PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
            else:
                if is_typeddict(schema):
                    response_format = convert_to_json_schema(schema)
                    if "required" not in response_format:
                        response_format["required"] = list(
                            response_format["properties"].keys()
                        )
                else:
                    # is JSON schema
                    response_format = cast(dict, schema)
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": response_format,
                    },
                )
                output_parser = JsonOutputParser()
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_schema', or 'json_mode'. Received: '{method}'"
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
