"""Ollama chat models."""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)
from uuid import uuid4

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from ollama import AsyncClient, Client, Message, Options
from pydantic import BaseModel, PrivateAttr, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self, is_typeddict

from ._utils import validate_model

log = logging.getLogger(__name__)


def _get_usage_metadata_from_generation_info(
    generation_info: Optional[Mapping[str, Any]],
) -> Optional[UsageMetadata]:
    """Get usage metadata from ollama generation info mapping."""
    if generation_info is None:
        return None
    input_tokens: Optional[int] = generation_info.get("prompt_eval_count")
    output_tokens: Optional[int] = generation_info.get("eval_count")
    if input_tokens is not None and output_tokens is not None:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    return None


def _parse_json_string(
    json_string: str,
    *,
    raw_tool_call: dict[str, Any],
    skip: bool,
) -> Any:
    """Attempt to parse a JSON string for tool calling.

    It first tries to use the standard json.loads. If that fails, it falls
    back to ast.literal_eval to safely parse Python literals, which is more
    robust against models using single quotes or containing apostrophes.

    Args:
        json_string: JSON string to parse.
        raw_tool_call: Raw tool call to include in error message.
        skip: Whether to ignore parsing errors and return the value anyways.

    Returns:
        The parsed JSON string or Python literal.

    Raises:
        OutputParserException: If the string is invalid and skip=False.
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
    See https://github.com/ollama/ollama/issues/6155
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


def _get_tool_calls_from_response(
    response: Mapping[str, Any],
) -> list[ToolCall]:
    """Get tool calls from ollama response."""
    tool_calls = []
    if "message" in response and (
        raw_tool_calls := response["message"].get("tool_calls")
    ):
        tool_calls.extend(
            [
                tool_call(
                    id=str(uuid4()),
                    name=tc["function"]["name"],
                    args=_parse_arguments_from_tool_call(tc) or {},
                )
                for tc in raw_tool_calls
            ]
        )
    return tool_calls


def _lc_tool_call_to_openai_tool_call(tool_call_: ToolCall) -> dict:
    """Convert a LangChain tool call to an OpenAI tool call format."""
    return {
        "type": "function",
        "id": tool_call_["id"],
        "function": {
            "name": tool_call_["name"],
            "arguments": tool_call_["args"],
        },
    }


def _get_image_from_data_content_block(block: dict) -> str:
    """Format standard data content block to format expected by Ollama."""
    if block["type"] == "image":
        if block["source_type"] == "base64":
            return block["data"]
        error_message = "Image data only supported through in-line base64 format."
        raise ValueError(error_message)

    error_message = f"Blocks of type {block['type']} not supported."
    raise ValueError(error_message)


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class ChatOllama(BaseChatModel):
    r"""Ollama chat model integration.

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
              captured and returned separately in the ``additional_kwargs`` of the
              response message, under ``reasoning_content``. The main response
              content will not include the reasoning tags.
            - ``False``: Disables reasoning mode. The model will not perform any reasoning,
              and the response will not include any reasoning content.
            - ``None`` (Default): The model will use its default reasoning behavior. Note
              however, if the model's default behavior *is* to perform reasoning, think tags
              (``<think>`` and ``</think>``) will be present within the main response content
              unless you set ``reasoning`` to ``True``.
        temperature: float
            Sampling temperature. Ranges from ``0.0`` to ``1.0``.
        num_predict: Optional[int]
            Max number of tokens to generate.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model = "llama3",
                temperature = 0.8,
                num_predict = 256,
                # other params ...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='J'adore le programmation. (Note: "programming" can also refer to the act of writing code, so if you meant that, I could translate it as "J'adore programmer". But since you didn\'t specify, I assumed you were talking about the activity itself, which is what "le programmation" usually refers to.)', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:37:50.182604Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 3576619666, 'load_duration': 788524916, 'prompt_eval_count': 32, 'prompt_eval_duration': 128125000, 'eval_count': 71, 'eval_duration': 2656556000}, id='run-ba48f958-6402-41a5-b461-5e250a4ebd36-0')

    Stream:
        .. code-block:: python

            messages = [
                ("human", "Return the words Hello World!"),
            ]
            for chunk in llm.stream(messages):
                print(chunk.text(), end="")


        .. code-block:: python

            content='Hello' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
            content=' World' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
            content='!' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
            content='' response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:39:42.274449Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 411875125, 'load_duration': 1898166, 'prompt_eval_count': 14, 'prompt_eval_duration': 297320000, 'eval_count': 4, 'eval_duration': 111099000} id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'


        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content='Je adore le programmation.(Note: "programmation" is the formal way to say "programming" in French, but informally, people might use the phrase "le développement logiciel" or simply "le code")', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:38:54.933154Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 1977300042, 'load_duration': 1345709, 'prompt_eval_duration': 159343000, 'eval_count': 47, 'eval_duration': 1815123000}, id='run-3c81a3ed-3e79-4dd3-a796-04064d804890')

    Async:
        .. code-block:: python

            messages = [
                ("human", "Hello how are you!"),
            ]
            await llm.ainvoke(messages)

        .. code-block:: python

            AIMessage(content="Hi there! I'm just an AI, so I don't have feelings or emotions like humans do. But I'm functioning properly and ready to help with any questions or tasks you may have! How can I assist you today?", response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:52:08.165478Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 2138492875, 'load_duration': 1364000, 'prompt_eval_count': 10, 'prompt_eval_duration': 297081000, 'eval_count': 47, 'eval_duration': 1838524000}, id='run-29c510ae-49a4-4cdd-8f23-b972bfab1c49-0')

        .. code-block:: python

            messages = [
                ("human", "Say hello world!"),
            ]
            async for chunk in llm.astream(messages):
                print(chunk.content)

        .. code-block:: python

            HEL
            LO
            WORLD
            !

        .. code-block:: python

            messages = [
                ("human", "Say hello world!"),
                ("human","Say goodbye world!")
            ]
            await llm.abatch(messages)

        .. code-block:: python

            [AIMessage(content='HELLO, WORLD!', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:55:07.315396Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 1696745458, 'load_duration': 1505000, 'prompt_eval_count': 8, 'prompt_eval_duration': 111627000, 'eval_count': 6, 'eval_duration': 185181000}, id='run-da6c7562-e25a-4a44-987a-2c83cd8c2686-0'),
            AIMessage(content="It's been a blast chatting with you! Say goodbye to the world for me, and don't forget to come back and visit us again soon!", response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:55:07.018076Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 1399391083, 'load_duration': 1187417, 'prompt_eval_count': 20, 'prompt_eval_duration': 230349000, 'eval_count': 31, 'eval_duration': 1166047000}, id='run-96cad530-6f3e-4cf9-86b4-e0f8abba4cdb-0')]

    JSON mode:
        .. code-block:: python


            json_llm = ChatOllama(format="json")
            messages = [
                ("human", "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. Respond using JSON only."),
            ]
            llm.invoke(messages).content

        .. code-block:: python

            '{"location": "Pune, India", "time_of_day": "morning"}'

    Tool Calling:
        .. code-block:: python

            from langchain_ollama import ChatOllama
            from pydantic import BaseModel, Field

            class Multiply(BaseModel):
                a: int = Field(..., description="First integer")
                b: int = Field(..., description="Second integer")

            ans = await chat.invoke("What is 45*67")
            ans.tool_calls

        .. code-block:: python

            [{'name': 'Multiply',
            'args': {'a': 45, 'b': 67},
            'id': '420c3f3b-df10-4188-945f-eb3abdb40622',
            'type': 'tool_call'}]

    Thinking / Reasoning:
        You can enable reasoning mode for models that support it by setting
        the ``reasoning`` parameter to ``True`` in either the constructor or
        the ``invoke``/``stream`` methods. This will enable the model to think
        through the problem and return the reasoning process separately in the
        ``additional_kwargs`` of the response message, under ``reasoning_content``.

        If ``reasoning`` is set to ``None``, the model will use its default reasoning
        behavior, and any reasoning content will *not* be captured under the
        ``reasoning_content`` key, but will be present within the main response content
        as think tags (``<think>`` and ``</think>``).

        .. note::
            This feature is only available for `models that support reasoning <https://ollama.com/search?c=thinking>`__.

        .. code-block:: python

            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model = "deepseek-r1:8b",
                reasoning= True,
            )

            user_message = HumanMessage(content="how many r in the word strawberry?")
            messages: List[Any] = [user_message]
            llm.invoke(messages)

            # or, on an invocation basis:

            llm.invoke(messages, reasoning=True)
            # or llm.stream(messages, reasoning=True)

            # If not provided, the invocation will default to the ChatOllama reasoning
            # param provided (None by default).

        .. code-block:: python

            AIMessage(content='The word "strawberry" contains **three \'r\' letters**. Here\'s a breakdown for clarity:\n\n- The spelling of "strawberry" has two parts ... be 3.\n\nTo be thorough, let\'s confirm with an online source or common knowledge.\n\nI can recall that "strawberry" has: s-t-r-a-w-b-e-r-r-y — yes, three r\'s.\n\nPerhaps it\'s misspelled by some, but standard is correct.\n\nSo I think the response should be 3.\n'}, response_metadata={'model': 'deepseek-r1:8b', 'created_at': '2025-07-08T19:33:55.891269Z', 'done': True, 'done_reason': 'stop', 'total_duration': 98232561292, 'load_duration': 28036792, 'prompt_eval_count': 10, 'prompt_eval_duration': 40171834, 'eval_count': 3615, 'eval_duration': 98163832416, 'model_name': 'deepseek-r1:8b'}, id='run--18f8269f-6a35-4a7c-826d-b89d52c753b3-0', usage_metadata={'input_tokens': 10, 'output_tokens': 3615, 'total_tokens': 3625})


    """  # noqa: E501, pylint: disable=line-too-long

    model: str
    """Model name to use."""

    reasoning: Optional[bool] = None
    """Controls the reasoning/thinking mode for
    `supported models <https://ollama.com/search?c=thinking>`__.

    - ``True``: Enables reasoning mode. The model's reasoning process will be
      captured and returned separately in the ``additional_kwargs`` of the
      response message, under ``reasoning_content``. The main response
      content will not include the reasoning tags.
    - ``False``: Disables reasoning mode. The model will not perform any reasoning,
      and the response will not include any reasoning content.
    - ``None`` (Default): The model will use its default reasoning behavior. Note
      however, if the model's default behavior *is* to perform reasoning, think tags
      ()``<think>`` and ``</think>``) will be present within the main response content
      unless you set ``reasoning`` to ``True``."""

    validate_model_on_init: bool = False
    """Whether to validate the model exists in Ollama locally on initialization.

    .. versionadded:: 0.3.4
    """

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict when generating text.
    (Default: 128, -1 = infinite generation, -2 = fill context)"""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    seed: Optional[int] = None
    """Sets the random number seed to use for generation. Setting this
    to a specific number will make the model generate the same text for
    the same prompt."""

    stop: Optional[list[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None
    """Specify the format of the output (options: "json", JSON schema)."""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx clients.
    These arguments are passed to both synchronous and async clients.
    Use sync_client_kwargs and async_client_kwargs to pass different arguments
    to synchronous and asynchronous clients.
    """

    async_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before
    passing to the httpx AsyncClient.
    `Full list of params. <https://www.python-httpx.org/api/#asyncclient>`__
    """

    sync_client_kwargs: Optional[dict] = {}
    """Additional kwargs to merge with client_kwargs before
    passing to the httpx Client.
    `Full list of params. <https://www.python-httpx.org/api/#client>`__
    """

    _client: Client = PrivateAttr()
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = PrivateAttr()
    """
    The async client to use for making requests.
    """

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ollama_messages = self._convert_messages_to_ollama_messages(messages)

        if self.stop is not None and stop is not None:
            msg = "`stop` found in both the input and default params."
            raise ValueError(msg)
        if self.stop is not None:
            stop = self.stop

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
                "stop": self.stop if stop is None else stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        )

        params = {
            "messages": ollama_messages,
            "stream": kwargs.pop("stream", True),
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

    def _convert_messages_to_ollama_messages(
        self, messages: list[BaseMessage]
    ) -> Sequence[Message]:
        ollama_messages: list = []
        for message in messages:
            role: str
            tool_call_id: Optional[str] = None
            tool_calls: Optional[list[dict[str, Any]]] = None
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
                tool_calls = (
                    [
                        _lc_tool_call_to_openai_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]
                    if message.tool_calls
                    else None
                )
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, ChatMessage):
                role = message.role
            elif isinstance(message, ToolMessage):
                role = "tool"
                tool_call_id = message.tool_call_id
            else:
                msg = "Received unsupported message type for Ollama."
                raise ValueError(msg)

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(list[dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "tool_use":
                        continue
                    elif content_part.get("type") == "image_url":
                        image_url = None
                        temp_image_url = content_part.get("image_url")
                        if isinstance(temp_image_url, str):
                            image_url = temp_image_url
                        elif (
                            isinstance(temp_image_url, dict)
                            and "url" in temp_image_url
                            and isinstance(temp_image_url["url"], str)
                        ):
                            image_url = temp_image_url["url"]
                        else:
                            msg = (
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )
                            raise ValueError(msg)

                        image_url_components = image_url.split(",")
                        # Support data:image/jpeg;base64,<image> format
                        # and base64 strings
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])
                    elif is_data_content_block(content_part):
                        image = _get_image_from_data_content_block(content_part)
                        images.append(image)
                    else:
                        msg = (
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )
                        raise ValueError(msg)
            # Should convert to ollama.Message once role includes tool,
            # and tool_call_id is in Message
            msg_: dict = {
                "role": role,
                "content": content,
                "images": images,
            }
            if tool_calls:
                msg_["tool_calls"] = tool_calls
            if tool_call_id:
                msg_["tool_call_id"] = tool_call_id
            ollama_messages.append(msg_)

        return ollama_messages

    async def _acreate_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[Mapping[str, Any], str]]:
        chat_params = self._chat_params(messages, stop, **kwargs)

        if chat_params["stream"]:
            async for part in await self._async_client.chat(**chat_params):
                yield part
        else:
            yield await self._async_client.chat(**chat_params)

    def _create_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        chat_params = self._chat_params(messages, stop, **kwargs)

        if chat_params["stream"]:
            if self._client:
                yield from self._client.chat(**chat_params)
        else:
            if self._client:
                yield self._client.chat(**chat_params)

    def _chat_stream_with_aggregation(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        for chunk in self._iterate_over_stream(messages, stop, **kwargs):
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        async for chunk in self._aiterate_over_stream(messages, stop, **kwargs):
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        return final_chunk

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="ollama",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = self._chat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        generation_info = final_chunk.generation_info
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=final_chunk.text,
                usage_metadata=cast(AIMessageChunk, final_chunk.message).usage_metadata,
                tool_calls=cast(AIMessageChunk, final_chunk.message).tool_calls,
                additional_kwargs=final_chunk.message.additional_kwargs,
            ),
            generation_info=generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _iterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                content = (
                    stream_resp["message"]["content"]
                    if "message" in stream_resp and "content" in stream_resp["message"]
                    else ""
                )

                # Warn and skip responses with done_reason: 'load' and empty content
                # These indicate the model was loaded but no actual generation occurred
                is_load_response_with_empty_content = (
                    stream_resp.get("done") is True
                    and stream_resp.get("done_reason") == "load"
                    and not content.strip()
                )

                if is_load_response_with_empty_content:
                    log.warning(
                        "Ollama returned empty response with done_reason='load'."
                        "This typically indicates the model was loaded but no content "
                        "was generated. Skipping this response."
                    )
                    continue

                if stream_resp.get("done") is True:
                    generation_info = dict(stream_resp)
                    if "model" in generation_info:
                        generation_info["model_name"] = generation_info["model"]
                    _ = generation_info.pop("message", None)
                else:
                    generation_info = None

                additional_kwargs = {}
                if (
                    reasoning
                    and "message" in stream_resp
                    and (thinking_content := stream_resp["message"].get("thinking"))
                ):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=generation_info,
                )

                yield chunk

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._iterate_over_stream(messages, stop, **kwargs):
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk

    async def _aiterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                content = (
                    stream_resp["message"]["content"]
                    if "message" in stream_resp and "content" in stream_resp["message"]
                    else ""
                )

                # Warn and skip responses with done_reason: 'load' and empty content
                # These indicate the model was loaded but no actual generation occurred
                is_load_response_with_empty_content = (
                    stream_resp.get("done") is True
                    and stream_resp.get("done_reason") == "load"
                    and not content.strip()
                )

                if is_load_response_with_empty_content:
                    log.warning(
                        "Ollama returned empty response with done_reason='load'. "
                        "This typically indicates the model was loaded but no content "
                        "was generated. Skipping this response."
                    )
                    continue

                if stream_resp.get("done") is True:
                    generation_info = dict(stream_resp)
                    if "model" in generation_info:
                        generation_info["model_name"] = generation_info["model"]
                    _ = generation_info.pop("message", None)
                else:
                    generation_info = None

                additional_kwargs = {}
                if (
                    reasoning
                    and "message" in stream_resp
                    and (thinking_content := stream_resp["message"].get("thinking"))
                ):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=generation_info,
                )

                yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._aiterate_over_stream(messages, stop, **kwargs):
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = await self._achat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        generation_info = final_chunk.generation_info
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=final_chunk.text,
                usage_metadata=cast(AIMessageChunk, final_chunk.message).usage_metadata,
                tool_calls=cast(AIMessageChunk, final_chunk.message).tool_calls,
                additional_kwargs=final_chunk.message.additional_kwargs,
            ),
            generation_info=generation_info,
        )
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ollama"

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"], bool]] = None,  # noqa: PYI051
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: If provided, which tool for model to call. **This parameter
                is currently ignored as it is not supported by Ollama.**
            kwargs: Any additional parameters are passed directly to
                ``self.bind(**kwargs)``.
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
                    Specifies ``format="json"``. Note that if using JSON mode then you
                    must include instructions for formatting the output into the
                    desired schema into the model call.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys ``'raw'``, ``'parsed'``, and ``'parsing_error'``.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs an instance of ``schema`` (i.e., a Pydantic object). Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:

            - ``'raw'``: BaseMessage
            - ``'parsed'``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - ``'parsing_error'``: Optional[BaseException]

        .. versionchanged:: 0.2.2

            Added support for structured output API via ``format`` parameter.

        .. versionchanged:: 0.3.0

            Updated default ``method`` to ``'json_schema'``.

        .. dropdown:: Example: schema=Pydantic class, method="json_schema", include_raw=False

            .. code-block:: python

                from typing import Optional

                from langchain_ollama import ChatOllama
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Optional[str] = Field(
                        default=..., description="A justification for the answer."
                    )


                llm = ChatOllama(model="llama3.1", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        .. dropdown:: Example: schema=Pydantic class, method="json_schema", include_raw=True

            .. code-block:: python

                from langchain_ollama import ChatOllama
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatOllama(model="llama3.1", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        .. dropdown:: Example: schema=Pydantic class, method="function_calling", include_raw=False

            .. code-block:: python

                from typing import Optional

                from langchain_ollama import ChatOllama
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Optional[str] = Field(
                        default=..., description="A justification for the answer."
                    )


                llm = ChatOllama(model="llama3.1", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, method="function_calling"
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        .. dropdown:: Example: schema=TypedDict class, method="function_calling", include_raw=False

            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_ollama import ChatOllama


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[
                        Optional[str], None, "A justification for the answer."
                    ]


                llm = ChatOllama(model="llama3.1", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        .. dropdown:: Example: schema=OpenAI function schema, method="function_calling", include_raw=False

            .. code-block:: python

                from langchain_ollama import ChatOllama

                oai_schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
               }

                llm = ChatOllama(model="llama3.1", temperature=0)
                structured_llm = llm.with_structured_output(oai_schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        .. dropdown:: Example: schema=Pydantic class, method="json_mode", include_raw=True

            .. code-block::

                from langchain_ollama import ChatOllama
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatOllama(model="llama3.1", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                #     'parsing_error': None
                # }
        """  # noqa: E501, D301
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
