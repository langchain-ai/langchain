"""Ollama chat models."""

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
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from ollama import AsyncClient, Client, Message, Options
from pydantic import PrivateAttr, model_validator
from typing_extensions import Self


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
    json_string: str, raw_tool_call: dict[str, Any], skip: bool
) -> Any:
    """Attempt to parse a JSON string for tool calling.

    Args:
        json_string: JSON string to parse.
        skip: Whether to ignore parsing errors and return the value anyways.
        raw_tool_call: Raw tool call to include in error message.

    Returns:
        The parsed JSON string.

    Raises:
        OutputParserException: If the JSON string wrong invalid and skip=False.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        if skip:
            return json_string
        msg = (
            f"Function {raw_tool_call['function']['name']} arguments:\n\n"
            f"{raw_tool_call['function']['arguments']}\n\nare not valid JSON. "
            f"Received JSONDecodeError {e}"
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
    parsed_arguments = {}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if isinstance(value, str):
                parsed_arguments[key] = _parse_json_string(
                    value, skip=True, raw_tool_call=raw_tool_call
                )
            else:
                parsed_arguments[key] = value
    else:
        parsed_arguments = _parse_json_string(
            arguments, skip=False, raw_tool_call=raw_tool_call
        )
    return parsed_arguments


def _get_tool_calls_from_response(
    response: Mapping[str, Any],
) -> List[ToolCall]:
    """Get tool calls from ollama response."""
    tool_calls = []
    if "message" in response:
        if raw_tool_calls := response["message"].get("tool_calls"):
            for tc in raw_tool_calls:
                tool_calls.append(
                    tool_call(
                        id=str(uuid4()),
                        name=tc["function"]["name"],
                        args=_parse_arguments_from_tool_call(tc) or {},
                    )
                )
    return tool_calls


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": tool_call["args"],
        },
    }


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
        temperature: float
            Sampling temperature. Ranges from 0.0 to 1.0.
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
                print(chunk)


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
        .. warning::
            Ollama currently does not support streaming for tools

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
    """  # noqa: E501

    model: str
    """Model name to use."""

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

    stop: Optional[List[str]] = None
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

    format: Literal["", "json"] = ""
    """Specify the format of the output (options: json)"""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx Client.
    For a full list of the params, see [this link](https://pydoc.dev/httpx/latest/httpx.Client.html)
    """

    _client: Client = PrivateAttr(default=None)  # type: ignore
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = PrivateAttr(default=None)  # type: ignore
    """
    The async client to use for making requests.
    """

    def _chat_params(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ollama_messages = self._convert_messages_to_ollama_messages(messages)

        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
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

        tools = kwargs.get("tools")
        default_stream = not bool(tools)

        params = {
            "messages": ollama_messages,
            "stream": kwargs.pop("stream", default_stream),
            "model": kwargs.pop("model", self.model),
            "format": kwargs.pop("format", self.format),
            "options": Options(**options_dict),
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

        if tools:
            params["tools"] = tools

        return params

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients to use for ollama."""
        client_kwargs = self.client_kwargs or {}
        self._client = Client(host=self.base_url, **client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **client_kwargs)
        return self

    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> Sequence[Message]:
        ollama_messages: List = []
        for message in messages:
            role: Literal["user", "assistant", "system", "tool"]
            tool_call_id: Optional[str] = None
            tool_calls: Optional[List[Dict[str, Any]]] = None
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
            elif isinstance(message, ToolMessage):
                role = "tool"
                tool_call_id = message.tool_call_id
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
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
                            raise ValueError(
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )

                        image_url_components = image_url.split(",")
                        # Support data:image/jpeg;base64,<image> format
                        # and base64 strings
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])

                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )
            # Should convert to ollama.Message once role includes tool, and tool_call_id is in Message # noqa: E501
            msg: dict = {
                "role": role,
                "content": content,
                "images": images,
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls  # type: ignore
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            ollama_messages.append(msg)

        return ollama_messages

    async def _acreate_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        chat_params = self._chat_params(messages, stop, **kwargs)

        if chat_params["stream"]:
            yield from self._client.chat(**chat_params)
        else:
            yield self._client.chat(**chat_params)

    def _chat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=(
                            stream_resp["message"]["content"]
                            if "message" in stream_resp
                            and "content" in stream_resp["message"]
                            else ""
                        ),
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
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
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=(
                            stream_resp["message"]["content"]
                            if "message" in stream_resp
                            and "content" in stream_resp["message"]
                            else ""
                        ),
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
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
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
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
            ),
            generation_info=generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=(
                            stream_resp["message"]["content"]
                            if "message" in stream_resp
                            and "content" in stream_resp["message"]
                            else ""
                        ),
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=(
                            stream_resp["message"]["content"]
                            if "message" in stream_resp
                            and "content" in stream_resp["message"]
                            else ""
                        ),
                        usage_metadata=_get_usage_metadata_from_generation_info(
                            stream_resp
                        ),
                        tool_calls=_get_tool_calls_from_response(stream_resp),
                    ),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
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
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"], bool]] = None,
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
        """  # noqa: E501
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
