"""Cloudflare Workers AI Chat wrapper."""

from __future__ import annotations

import json
import uuid
import warnings
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
)

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
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    from_env,
    get_pydantic_field_names,
    secret_from_env,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

# from langchain_cloudflare.version import __version__
__version__ = "0.0.1"


class ChatCloudflareWorkersAI(BaseChatModel):
    """`Cloudflare Workers AI` Chat large language models API.

    To use, you should have the
    environment variables ``CLOUDFLARE_API_TOKEN`` and ``CLOUDFLARE_ACCOUNT_ID``
    set with your API token and account ID.

    Any parameters that are valid to be passed to the Cloudflare Workers AI API call
    can be passed in, even if not explicitly saved on this class.

    Setup:
        Install ``langchain-cloudflare`` and set environment variables:

        .. code-block:: bash

            pip install -U langchain-cloudflare
            export CLOUDFLARE_API_TOKEN="your-api-token"
            export CLOUDFLARE_ACCOUNT_ID="your-account-id"

    Key init args — completion params:
        model: str
            Name of Cloudflare Workers AI model to use. E.g.
            "@cf/meta/llama-3.1-8b-instruct".
        temperature: float
            Sampling temperature. Ranges from 0.0 to 1.0.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        model_kwargs: Dict[str, Any]
            Holds any model parameters valid for API call not
            explicitly specified.

    Key init args — client params:
        timeout: Union[float, Tuple[float, float], Any, None]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_token: Optional[str]
            Cloudflare API token. If not passed in will be read
            from env var CLOUDFLARE_API_TOKEN.
        account_id: Optional[str]
            Cloudflare account ID. If not passed in will be read
            from env var CLOUDFLARE_ACCOUNT_ID.
        base_url: Optional[str]
            Base URL path for API requests, leave blank if not using a proxy
            or service emulator.
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    stop: Optional[Union[List[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""
    cloudflare_api_token: Optional[SecretStr] = Field(
        alias="api_token",
        default_factory=secret_from_env("CLOUDFLARE_API_TOKEN", default=None),
    )
    """Automatically inferred from env var `CLOUDFLARE_API_TOKEN` if not provided."""
    cloudflare_account_id: Optional[str] = Field(
        alias="account_id",
        default_factory=from_env("CLOUDFLARE_ACCOUNT_ID", default=None),
    )
    """Automatically inferred from env var `CLOUDFLARE_ACCOUNT_ID` if not provided."""
    cloudflare_api_base: Optional[str] = Field(
        alias="base_url", default_factory=from_env("CLOUDFLARE_API_BASE", default=None)
    )
    """Base URL path for API requests, 
    leave blank if not using a proxy or service emulator."""
    # to support explicit proxy for Cloudflare
    cloudflare_proxy: Optional[str] = Field(
        default_factory=from_env("CLOUDFLARE_PROXY", default=None)
    )
    cloudflare_gateway: Optional[str] = Field(
        alias="ai_gateway",
        default_factory=from_env("CLOUDFLARE_AI_GATEWAY", default=None),
    )
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Cloudflare API. Can be float, httpx.Timeout or None."""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""
    http_async_client: Union[Any, None] = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify
        http_client as well if you'd like a custom client for sync invocations."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api token and account ID exist in environment."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")
        if self.temperature == 0:
            self.temperature = 1e-8

        if not self.cloudflare_api_token:
            raise ValueError(
                "A Cloudflare API token must be provided either through "
                "the api_token parameter or "
                "CLOUDFLARE_API_TOKEN environment variable."
            )

        if not self.cloudflare_account_id:
            raise ValueError(
                "A Cloudflare account ID must be provided either through "
                "the account_id parameter or "
                "CLOUDFLARE_ACCOUNT_ID environment variable."
            )

        default_headers = {"User-Agent": f"langchain/{__version__}"} | dict(
            self.default_headers or {}
        )

        try:
            import httpx

            # Determine the base URL based on whether
            # we're using AI Gateway or direct API
            if self.cloudflare_gateway:
                base_url = (
                    f"https://gateway.ai.cloudflare.com/v1/"
                    f"{self.cloudflare_account_id}/{self.cloudflare_gateway}"
                )
            else:
                # Use the custom base_url if provided, otherwise use the default
                base_url = (
                    self.cloudflare_api_base or "https://api.cloudflare.com/client/v4"
                )

            # Configure the httpx client
            if not self.client:
                self.client = httpx.Client(
                    base_url=base_url,
                    timeout=self.request_timeout,
                    headers={
                        "Authorization": f"Bearer "
                        f"{self.cloudflare_api_token.get_secret_value()}",
                        **default_headers,
                    },
                )

            if not self.async_client:
                self.async_client = httpx.AsyncClient(
                    base_url=base_url,
                    timeout=self.request_timeout,
                    headers={
                        "Authorization": f"Bearer "
                        f"{self.cloudflare_api_token.get_secret_value()}",
                        **default_headers,
                    },
                )

        except ImportError:
            raise ImportError(
                "Could not import httpx python package. "
                "Please install it with `pip install httpx`."
            )

        return self

    #
    # Serializable class method overrides
    #
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"cloudflare_api_token": "CLOUDFLARE_API_TOKEN"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    #
    # BaseChatModel method overrides
    #
    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "cloudflare-workers-ai-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="cloudflare-workers-ai",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    def _should_stream(
        self,
        *,
        async_api: bool,
        run_manager: Optional[
            Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]
        ] = None,
        **kwargs: Any,
    ) -> bool:
        """Determine if a given model call should hit the streaming API."""
        base_should_stream = super()._should_stream(
            async_api=async_api, run_manager=run_manager, **kwargs
        )
        if base_should_stream and ("response_format" in kwargs):
            # Streaming not supported in JSON mode.
            return kwargs["response_format"] != {"type": "json_object"}
        return base_should_stream

    def _extract_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from content if it appears to be a JSON tool call."""
        tool_calls = []

        # Only attempt to parse if content looks like JSON
        if (
            not content
            or not isinstance(content, str)
            or not (
                (content.startswith("{") and content.endswith("}"))
                or (content.startswith("[") and content.endswith("]"))
            )
        ):
            return tool_calls

        try:
            content_json = json.loads(content)

            # Handle direct top-level tool_calls array format
            if isinstance(content_json, dict) and "tool_calls" in content_json:
                for tool_call in content_json["tool_calls"]:
                    if (
                        isinstance(tool_call, dict)
                        and "name" in tool_call
                        and "arguments" in tool_call
                    ):
                        tool_id = tool_call.get("id", f"call_{uuid.uuid4()}")
                        tool_calls.append(
                            {
                                "id": tool_id,
                                "name": tool_call["name"],
                                "args": tool_call["arguments"],
                                "type": "function",
                            }
                        )
                return tool_calls

            # Check for result.tool_calls format
            if isinstance(content_json, dict) and "result" in content_json:
                result = content_json["result"]
                if isinstance(result, dict) and "tool_calls" in result:
                    for tool_call in result["tool_calls"]:
                        if (
                            isinstance(tool_call, dict)
                            and "name" in tool_call
                            and "arguments" in tool_call
                        ):
                            tool_id = tool_call.get("id", f"call_{uuid.uuid4()}")
                            tool_calls.append(
                                {
                                    "id": tool_id,
                                    "name": tool_call["name"],
                                    "args": tool_call["arguments"],
                                    "type": "function",
                                }
                            )
                    return tool_calls

            # Handle single tool call format
            if isinstance(content_json, dict):
                # Direct tool call format
                if "name" in content_json and (
                    "arguments" in content_json or "parameters" in content_json
                ):
                    args = content_json.get(
                        "arguments", content_json.get("parameters", {})
                    )
                    tool_id = content_json.get("id", f"call_{uuid.uuid4()}")
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "name": content_json["name"],
                            "args": args,
                            "type": "function",
                        }
                    )
                    return tool_calls

                # OpenAI format with function property
                if "function" in content_json and isinstance(
                    content_json["function"], dict
                ):
                    name = content_json["function"].get("name")
                    args_str = content_json["function"].get("arguments", "{}")

                    if name:
                        try:
                            args = (
                                json.loads(args_str)
                                if isinstance(args_str, str)
                                else args_str
                            )
                            tool_calls.append(
                                {
                                    "id": content_json.get(
                                        "id", f"call_{uuid.uuid4()}"
                                    ),
                                    "name": name,
                                    "args": args,
                                    "type": "function",
                                }
                            )
                        except json.JSONDecodeError:
                            tool_calls.append(
                                {
                                    "id": content_json.get(
                                        "id", f"call_{uuid.uuid4()}"
                                    ),
                                    "name": name,
                                    "args": {"raw_args": args_str},
                                    "type": "function",
                                }
                            )
                        return tool_calls

            # Handle multiple tool calls in an array
            if isinstance(content_json, list):
                for item in content_json:
                    if isinstance(item, dict):
                        # Direct format
                        if "name" in item and (
                            "arguments" in item or "parameters" in item
                        ):
                            args = item.get("arguments", item.get("parameters", {}))
                            tool_id = item.get("id", f"call_{uuid.uuid4()}")
                            tool_calls.append(
                                {
                                    "id": tool_id,
                                    "name": item["name"],
                                    "args": args,
                                    "type": "function",
                                }
                            )
                        # OpenAI format
                        elif "function" in item and isinstance(item["function"], dict):
                            name = item["function"].get("name")
                            args_str = item["function"].get("arguments", "{}")

                            if name:
                                try:
                                    args = (
                                        json.loads(args_str)
                                        if isinstance(args_str, str)
                                        else args_str
                                    )
                                    tool_calls.append(
                                        {
                                            "id": item.get(
                                                "id", f"call_{uuid.uuid4()}"
                                            ),
                                            "name": name,
                                            "args": args,
                                            "type": "function",
                                        }
                                    )
                                except json.JSONDecodeError:
                                    tool_calls.append(
                                        {
                                            "id": item.get(
                                                "id", f"call_{uuid.uuid4()}"
                                            ),
                                            "name": name,
                                            "args": {"raw_args": args_str},
                                            "type": "function",
                                        }
                                    )
        except json.JSONDecodeError:
            # Not valid or complete JSON yet
            pass

        return tool_calls

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

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
        }

        # Construct the API URL
        if self.cloudflare_gateway:
            # If using AI Gateway
            api_url = f"workers-ai/run/{self.model_name}"
        else:
            # If using direct API
            api_url = f"accounts/{self.cloudflare_account_id}/ai/run/{self.model_name}"

        # Create the request payload
        payload = {"messages": message_dicts, **params}

        # Make the API request
        response = self.client.post(api_url, json=payload)
        response.raise_for_status()
        response_data = response.json()

        return self._create_chat_result(response_data)

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

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
        }

        # Construct the Cloudflare Workers AI API URL
        if self.cloudflare_gateway:
            api_url = f"workers-ai/run/{self.model_name}"
        else:
            api_url = f"accounts/{self.cloudflare_account_id}/ai/run/{self.model_name}"

        # Create the request payload
        payload = {"messages": message_dicts, **params}

        # Make the API request
        response = await self.async_client.post(api_url, json=payload)
        response.raise_for_status()
        response_data = response.json()

        return self._create_chat_result(response_data)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream completion for Cloudflare Workers AI."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        # Construct the Cloudflare Workers AI API URL
        if self.cloudflare_gateway:
            api_url = f"workers-ai/run/{self.model_name}"
        else:
            api_url = f"accounts/{self.cloudflare_account_id}/ai/run/{self.model_name}"

        # Create the request payload
        payload = {"messages": message_dicts, **params}

        # Make the streaming API request
        with self.client.stream("POST", api_url, json=payload) as response:
            response.raise_for_status()
            accumulated_content = ""
            tool_calls_detected = False

            for line in response.iter_lines():
                # Handle both string and bytes cases
                if isinstance(line, bytes):
                    # Convert bytes to string if needed
                    line_str = line.decode("utf-8")
                else:
                    line_str = line

                if not line_str or line_str.startswith("data: [DONE]"):
                    continue

                if line_str.startswith("data: "):
                    line_str = line_str[6:]  # Remove "data: " prefix

                try:
                    chunk = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Process streaming response
                if "streamed_data" in chunk:
                    # Handle streamed_data format from Cloudflare
                    for stream_chunk in chunk["streamed_data"]:
                        response_text = stream_chunk.get("response", "")
                        accumulated_content += response_text

                        # Initialize generation_info
                        generation_info = {}
                        if "usage" in stream_chunk:
                            generation_info["usage"] = stream_chunk["usage"]

                        # Check for tool calls in accumulated content
                        tool_calls = []

                        # Only try to extract tool calls
                        # if we haven't already detected them
                        if not tool_calls_detected:
                            tool_calls = self._extract_tool_calls_from_content(
                                accumulated_content
                            )
                            if tool_calls:
                                tool_calls_detected = True

                        # Create the message chunk -
                        # if tool calls exist, don't include content
                        if tool_calls:
                            message_chunk = AIMessageChunk(
                                content=None,
                                tool_calls=tool_calls,
                            )
                        else:
                            message_chunk = AIMessageChunk(
                                content=response_text
                                if not tool_calls_detected
                                else "",
                            )

                        generation_chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info or None,
                        )

                        if run_manager:
                            run_manager.on_llm_new_token(
                                "" if tool_calls else generation_chunk.text,
                                chunk=generation_chunk,
                            )

                        yield generation_chunk

                elif "response" in chunk:
                    # Handle direct response format
                    response_text = chunk.get("response", "")
                    accumulated_content += response_text

                    # Check for tool calls
                    tool_calls = []

                    # Only try to extract tool calls if we haven't already detected them
                    if not tool_calls_detected:
                        # First check if there are explicit tool_calls in the response
                        if "tool_calls" in chunk:
                            for tc in chunk["tool_calls"]:
                                if "name" in tc and "arguments" in tc:
                                    tool_id = tc.get("id", str(uuid.uuid4()))
                                    tool_calls.append(
                                        {
                                            "id": tool_id,
                                            "name": tc["name"],
                                            "args": tc["arguments"],
                                            "type": "function",
                                        }
                                    )
                            tool_calls_detected = True
                        else:
                            # Try to extract from content
                            tool_calls = self._extract_tool_calls_from_content(
                                accumulated_content
                            )
                            if tool_calls:
                                tool_calls_detected = True

                    # Create message chunk based on whether we have tool calls
                    if tool_calls:
                        message_chunk = AIMessageChunk(
                            content=None,  # Use None, not empty string
                            tool_calls=tool_calls,
                        )
                    else:
                        message_chunk = AIMessageChunk(
                            content=response_text if not tool_calls_detected else "",
                        )

                    generation_info = {}
                    if "usage" in chunk:
                        generation_info["usage"] = chunk["usage"]

                    generation_chunk = ChatGenerationChunk(
                        message=message_chunk, generation_info=generation_info or None
                    )

                    if run_manager:
                        run_manager.on_llm_new_token(
                            "" if tool_calls else generation_chunk.text,
                            chunk=generation_chunk,
                        )

                    yield generation_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)

        params = {**params, **kwargs, "stream": True}

        # Construct the Cloudflare Workers AI API URL
        api_url = f"accounts/{self.cloudflare_account_id}/ai/run/{self.model_name}"

        # Create the request payload
        payload = {"messages": message_dicts, **params}

        # Make the streaming API request
        async with self.async_client.stream("POST", api_url, json=payload) as response:
            response.raise_for_status()

            accumulated_content = ""

            async for line in response.aiter_lines():
                # Handle both string and bytes cases
                if isinstance(line, bytes):
                    # Convert bytes to string if needed
                    line_str = line.decode("utf-8")
                else:
                    line_str = line

                if not line_str or line_str.startswith("data: [DONE]"):
                    continue

                if line_str.startswith("data: "):
                    line_str = line_str[6:]  # Remove "data: " prefix

                try:
                    chunk = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Handle the streamed_data format
                if "streamed_data" in chunk:
                    for stream_chunk in chunk["streamed_data"]:
                        response_text = stream_chunk.get("response", "")
                        accumulated_content += response_text

                        # Check if this is the final chunk with usage data
                        generation_info = {}
                        if "usage" in stream_chunk:
                            generation_info["usage"] = stream_chunk["usage"]

                        # Check for tool calls in accumulated content
                        tool_calls = []
                        try:
                            if (
                                accumulated_content.startswith("{")
                                and accumulated_content.endswith("}")
                            ) or (
                                accumulated_content.startswith("[")
                                and accumulated_content.endswith("]")
                            ):
                                content_json = json.loads(accumulated_content)
                                if (
                                    isinstance(content_json, dict)
                                    and "name" in content_json
                                    and "parameters" in content_json
                                ):
                                    # Found a tool call in JSON format
                                    tool_call_id = content_json.get(
                                        "id", f"call_{hash(str(content_json))}"
                                    )
                                    tool_calls = [
                                        {
                                            "id": tool_call_id,
                                            "name": content_json["name"],
                                            "args": content_json["parameters"],
                                        }
                                    ]
                        except json.JSONDecodeError:
                            # Not a complete JSON yet, continue streaming
                            pass

                        message_chunk = AIMessageChunk(
                            content=response_text if response_text is not None else "",
                            tool_calls=tool_calls,  # Always pass a list, even if empty
                        )

                        generation_chunk = ChatGenerationChunk(
                            message=message_chunk,
                            generation_info=generation_info or None,
                        )

                        if run_manager:
                            await run_manager.on_llm_new_token(
                                token=generation_chunk.text,
                                chunk=generation_chunk,
                            )

                        yield generation_chunk

                # Handle direct response format (as in previous implementation)
                elif "response" in chunk:
                    # Accumulate content to check for potential tool calls
                    response_text = chunk.get("response", "")
                    accumulated_content += response_text

                    # Check if the response is a complete
                    # JSON that might contain a tool call
                    tool_calls = []
                    try:
                        if (
                            accumulated_content.startswith("{")
                            and accumulated_content.endswith("}")
                        ) or (
                            accumulated_content.startswith("[")
                            and accumulated_content.endswith("]")
                        ):
                            content_json = json.loads(accumulated_content)
                            if (
                                isinstance(content_json, dict)
                                and "name" in content_json
                                and "parameters" in content_json
                            ):
                                # Found a tool call in JSON format
                                tool_call_id = content_json.get(
                                    "id", f"call_{hash(str(content_json))}"
                                )
                                tool_calls = [
                                    {
                                        "id": tool_call_id,
                                        "name": content_json["name"],
                                        "args": content_json["parameters"],
                                    }
                                ]
                                response_text = (
                                    ""  # Empty the response text since it's a tool call
                                )
                    except json.JSONDecodeError:
                        # Not a complete JSON yet, continue streaming
                        pass

                    message_chunk = AIMessageChunk(
                        content=response_text if response_text is not None else "",
                        tool_calls=tool_calls,  # Always pass a list, even if empty
                    )

                    generation_info = {}

                    # Include usage stats when available
                    if "usage" in chunk:
                        generation_info["usage"] = chunk["usage"]

                    generation_chunk = ChatGenerationChunk(
                        message=message_chunk, generation_info=generation_info or None
                    )

                    if run_manager:
                        await run_manager.on_llm_new_token(
                            token=generation_chunk.text,
                            chunk=generation_chunk,
                        )

                    yield generation_chunk

    #
    # Internal methods
    #
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cloudflare Workers AI API."""
        params = {
            "stream": self.streaming,
            "temperature": self.temperature,
            "stop": self.stop,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        generations = []

        # Extract the response data
        if "result" in response:
            response_result = response["result"]
        else:
            response_result = response

        content = response_result.get("response", "")
        token_usage = response_result.get("usage", {})
        tool_calls = []

        # Handle tool_calls directly from the response
        # First check for top-level tool_calls field as in the new format
        if "tool_calls" in response_result:
            raw_tool_calls = response_result["tool_calls"]
            for raw_tool_call in raw_tool_calls:
                if isinstance(raw_tool_call, dict):
                    # Handle format {"name": "...", "arguments": {...}}
                    if "name" in raw_tool_call and "arguments" in raw_tool_call:
                        tool_id = raw_tool_call.get("id", str(uuid.uuid4()))
                        tool_name = raw_tool_call["name"]
                        tool_args = raw_tool_call["arguments"]

                        tool_calls.append(
                            {
                                "id": tool_id,
                                "name": tool_name,
                                "args": tool_args,
                                "type": "function",
                            }
                        )
        # Try to parse tool calls from content if not explicitly provided
        # and content looks like JSON
        elif (
            content
            and isinstance(content, str)
            and (content.startswith("{") or content.startswith("["))
        ):
            try:
                content_json = json.loads(content)
                # Check for tool call pattern in content
                if isinstance(content_json, dict):
                    if "name" in content_json and "arguments" in content_json:
                        tool_id = content_json.get("id", str(uuid.uuid4()))
                        tool_calls.append(
                            {
                                "id": tool_id,
                                "name": content_json["name"],
                                "args": content_json["arguments"],
                                "type": "function",
                            }
                        )
                    # Check for nested result.tool_calls format
                    elif (
                        "result" in content_json
                        and isinstance(content_json["result"], dict)
                        and "tool_calls" in content_json["result"]
                    ):
                        for tc in content_json["result"]["tool_calls"]:
                            if (
                                isinstance(tc, dict)
                                and "name" in tc
                                and "arguments" in tc
                            ):
                                tool_id = tc.get("id", str(uuid.uuid4()))
                                tool_calls.append(
                                    {
                                        "id": tool_id,
                                        "name": tc["name"],
                                        "args": tc["arguments"],
                                        "type": "function",
                                    }
                                )
                # Handle array format
                elif isinstance(content_json, list):
                    for item in content_json:
                        if (
                            isinstance(item, dict)
                            and "name" in item
                            and "arguments" in item
                        ):
                            tool_id = item.get("id", str(uuid.uuid4()))
                            tool_calls.append(
                                {
                                    "id": tool_id,
                                    "name": item["name"],
                                    "args": item["arguments"],
                                    "type": "function",
                                }
                            )
            except json.JSONDecodeError:
                # Not JSON, treat as regular content
                pass

        # Create the AI message
        # When tool calls exist, set content to empty string
        if tool_calls:
            message = AIMessage(
                content="",  # Empty string instead of None to pass validation
                tool_calls=tool_calls,
            )
        else:
            # Use empty string if content is None
            if content is None:
                content = ""
            message = AIMessage(
                content=content,
            )

        # Add usage metadata
        if token_usage and isinstance(message, AIMessage):
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
            message.usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": token_usage.get(
                    "total_tokens", input_tokens + output_tokens
                ),
            }

        # Create generation and return result
        generation_info = {}
        gen = ChatGeneration(
            message=message,
            generation_info=generation_info,
        )
        generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Convert LangChain messages to Cloudflare Workers AI format.

        Improved to match the TypeScript implementation's format expectations.
        Different formatting for Mistral vs Llama models.
        """
        params = self._default_params
        if stop is not None:
            params["stop"] = stop

        # Check if this is a Llama model
        is_llama_model = "llama" in self.model_name.lower()

        # Convert messages to Cloudflare format
        cloudflare_messages = []

        for message in messages:
            # Base structure for each message
            msg: Dict[str, Any] = {}

            # Handle different message types
            if isinstance(message, ChatMessage):
                msg = {"role": message.role, "content": message.content}

            elif isinstance(message, HumanMessage):
                # Format message parts
                msg = {"role": "user", "content": message.content}

            elif isinstance(message, AIMessage):
                # Handle differently based on model type
                if message.tool_calls:
                    if is_llama_model:
                        # For Llama models -
                        # use content as JSON string representation of tool call
                        tool_call = message.tool_calls[0]
                        content_json = {
                            "name": tool_call["name"],
                            "parameters": tool_call["args"],
                        }
                        content_str = json.dumps(content_json)
                        msg = {"role": "assistant", "content": content_str}

                        # Also include the tool_calls in the standard format
                        tool_calls = []
                        for tc in message.tool_calls:
                            # Format args as JSON string
                            args_str = (
                                json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"]
                            )
                            tool_calls.append(
                                {
                                    "id": tc.get("id", str(uuid.uuid4())),
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": args_str,
                                    },
                                }
                            )
                        msg["tool_calls"] = tool_calls

                    else:
                        # For Mistral and other models -
                        # use the format with empty content
                        msg = {"role": "assistant", "content": ""}

                        # Format tool calls
                        tool_calls = []
                        for tc in message.tool_calls:
                            # Format args as JSON string
                            args_str = (
                                json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"]
                            )

                            tool_calls.append(
                                {
                                    "id": tc.get("id", f"call_{hash(str(tc))}"),
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": args_str,
                                    },
                                }
                            )

                        if tool_calls:
                            msg["tool_calls"] = tool_calls
                else:
                    # For regular assistant messages without tool calls
                    msg = {"role": "assistant", "content": message.content or ""}

                    # Handle legacy function_call format for backward compatibility
                    if "function_call" in message.additional_kwargs:
                        function_call = message.additional_kwargs["function_call"]
                        msg["function_call"] = function_call
                        if not is_llama_model:
                            # For non-Llama models,
                            # set content to empty when function_call exists
                            msg["content"] = ""

            elif isinstance(message, SystemMessage):
                msg = {"role": "system", "content": message.content}

            elif isinstance(message, FunctionMessage):
                msg = {
                    "role": "function",
                    "name": message.name,
                    "content": message.content,
                }

            elif isinstance(message, ToolMessage):
                msg = {"role": "tool", "name": message.name, "content": message.content}

                # Only include tool_call_id if it's available and needed
                if message.tool_call_id:
                    msg["tool_call_id"] = message.tool_call_id

            # Add any additional kwargs that might be needed
            for key, value in message.additional_kwargs.items():
                if key not in msg and key not in ["function_call", "tool_calls"]:
                    msg[key] = value

            cloudflare_messages.append(msg)

        return cloudflare_messages, params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage and v is not None:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        return combined

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        for tool in formatted_tools:
            # Ensure that all tools have the required schema fields
            if "$schema" not in tool["function"]["parameters"]:
                tool["function"]["parameters"]["$schema"] = (
                    "http://json-schema.org/draft-07/schema#"
                )

            # Set additionalProperties to false if not already set
            if "additionalProperties" not in tool["function"]["parameters"]:
                tool["function"]["parameters"]["additionalProperties"] = False

        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class (supported added in 0.1.9),
                    - or a Pydantic class.
                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.
            method:
                The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API.
                If "json_mode" then Cloudflare Workers AI JSON mode will be used.
            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes same inputs as a
            :class:`langchain_core.language_models.chat.BaseChatModel`.
        """
        _ = kwargs.pop("strict", None)
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "json_schema":
            # Some applications require that incompatible parameters (e.g., unsupported
            # methods) be handled.
            method = "function_calling"
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )

            # Custom parser for handling Cloudflare specific output format
            if is_pydantic_schema:

                class CloudflarePydanticToolsParser(PydanticToolsParser):
                    """Custom parser for handling Cloudflare Workers AI
                    tool outputs in various formats."""

                    def parse(self, message: BaseMessage) -> Any:
                        """Parse the message specifically for Cloudflare format."""
                        # First attempt: check tool_calls in the message
                        if message.tool_calls:
                            for tool_call in message.tool_calls:
                                if (
                                    self.tools
                                    and self.first_tool_only
                                    and len(self.tools) > 0
                                ):
                                    tool_class = self.tools[0]
                                    # Handle different args formats
                                    args = tool_call.get("args", {})

                                    # If args is a string, try to parse it as JSON first
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except json.JSONDecodeError:
                                            # Keep as is if not valid JSON
                                            args = {"raw_args": args}

                                    # Attempt to instantiate the model with the args
                                    try:
                                        # For Pydantic v2
                                        if hasattr(tool_class, "model_validate"):
                                            return tool_class.model_validate(args)
                                        # For Pydantic v1
                                        else:
                                            return tool_class.parse_obj(args)
                                    except Exception:
                                        # If first attempt fails,
                                        # continue to next approach
                                        pass

                        # Second attempt: check if content is
                        # JSON that contains tool call data
                        content = message.content
                        if content and isinstance(content, str):
                            # Only try parsing if it looks like JSON
                            if (content.startswith("{") and content.endswith("}")) or (
                                content.startswith("[") and content.endswith("[")
                            ):
                                try:
                                    content_json = json.loads(content)

                                    # Handle different formats
                                    # that Cloudflare might return
                                    if (
                                        self.tools
                                        and self.first_tool_only
                                        and len(self.tools) > 0
                                    ):
                                        tool_class = self.tools[0]

                                        # Try direct parsing first
                                        # (if content is the actual structure)
                                        try:
                                            # For Pydantic v2
                                            if hasattr(tool_class, "model_validate"):
                                                return tool_class.model_validate(
                                                    content_json
                                                )
                                            # For Pydantic v1
                                            else:
                                                return tool_class.parse_obj(
                                                    content_json
                                                )
                                        except Exception:
                                            # Check if it's wrapped
                                            # in a function call format
                                            # 1. Format:
                                            # {"name": "tool_name", "parameters": {...}}
                                            if (
                                                isinstance(content_json, dict)
                                                and "parameters" in content_json
                                            ):
                                                try:
                                                    params = content_json["parameters"]
                                                    if hasattr(
                                                        tool_class, "model_validate"
                                                    ):
                                                        return (
                                                            tool_class.model_validate(
                                                                params
                                                            )
                                                        )
                                                    else:
                                                        return tool_class.parse_obj(
                                                            params
                                                        )
                                                except Exception:
                                                    pass

                                            # 2. OpenAI format with function property
                                            if (
                                                isinstance(content_json, dict)
                                                and "function" in content_json
                                            ):
                                                try:
                                                    func_data = content_json["function"]
                                                    if "arguments" in func_data:
                                                        args = func_data["arguments"]
                                                        # Parse arguments
                                                        # if they're a string
                                                        if isinstance(args, str):
                                                            try:
                                                                args = json.loads(args)
                                                            except Exception:
                                                                args = {"raw": args}

                                                        if hasattr(
                                                            tool_class, "model_validate"
                                                        ):
                                                            return (
                                                                tool_class.model_validate(
                                                                    args
                                                                )
                                                            )
                                                        else:
                                                            return tool_class.parse_obj(
                                                                args
                                                            )
                                                except Exception:
                                                    pass

                                except json.JSONDecodeError:
                                    pass

                        # Last resort: try standard parsing
                        try:
                            return super().parse(message)
                        except Exception as e:
                            # Final fallback for complex cases
                            if (
                                self.tools
                                and self.first_tool_only
                                and len(self.tools) > 0
                            ):
                                tool_class = self.tools[0]

                                # Try different approaches based on API version
                                if isinstance(content, str) and content:
                                    try:
                                        # Try extracting JSON from string with regex
                                        import re

                                        json_match = re.search(
                                            r"{.*}", content, re.DOTALL
                                        )
                                        if json_match:
                                            json_str = json_match.group(0)
                                            try:
                                                data = json.loads(json_str)
                                                if hasattr(
                                                    tool_class, "model_validate"
                                                ):
                                                    return tool_class.model_validate(
                                                        data
                                                    )
                                                else:
                                                    return tool_class.parse_obj(data)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                            # If all attempts failed, raise the original error
                            raise e

                output_parser: OutputParserLike = CloudflarePydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:

                class CloudflareJsonOutputKeyToolsParser(JsonOutputKeyToolsParser):
                    """Custom parser for handling
                    Cloudflare Workers AI JSON tool outputs."""

                    def parse(self, message: BaseMessage) -> Any:
                        """Parse the message specifically for Cloudflare format."""
                        # First check tool_calls in the message (preferred path)
                        if message.tool_calls:
                            for tool_call in message.tool_calls:
                                if tool_call.get("name") == self.key_name:
                                    args = tool_call.get("args", {})
                                    # If args is a string, try parsing it
                                    if isinstance(args, str):
                                        try:
                                            return json.loads(args)
                                        except json.JSONDecodeError:
                                            return {"raw": args}
                                    return args

                        # Check if content has JSON that might contain our data
                        content = message.content
                        if content and isinstance(content, str):
                            # Only try parsing if it looks like JSON
                            if (content.startswith("{") and content.endswith("}")) or (
                                content.startswith("[") and content.endswith("]")
                            ):
                                try:
                                    content_json = json.loads(content)

                                    # Look for key in different formats
                                    # 1. Direct key access
                                    if isinstance(content_json, dict):
                                        if self.key_name in content_json:
                                            return content_json[self.key_name]

                                        # 2. Tool call format:
                                        # {"name": key_name, "parameters": {...}}
                                        if (
                                            "name" in content_json
                                            and content_json.get("name")
                                            == self.key_name
                                        ):
                                            return content_json.get("parameters", {})

                                        # 3. OpenAI format with function property
                                        if "function" in content_json:
                                            func_data = content_json["function"]
                                            if func_data.get("name") == self.key_name:
                                                args = func_data.get("arguments")
                                                if isinstance(args, str):
                                                    try:
                                                        return json.loads(args)
                                                    except Exception:
                                                        return {"raw": args}
                                                return args

                                    # 4. If it's not a specific format we recognize,
                                    # but content is a dict, return it as-is
                                    if isinstance(content_json, dict):
                                        return content_json

                                except json.JSONDecodeError:
                                    pass

                        # Try standard parsing
                        try:
                            return super().parse(message)
                        except Exception as e:
                            # Last resort: try to extract JSON from the content
                            if content and isinstance(content, str):
                                import re

                                json_match = re.search(r"{.*}", content, re.DOTALL)
                                if json_match:
                                    try:
                                        return json.loads(json_match.group(0))
                                    except Exception:
                                        pass
                            raise e

                output_parser = CloudflareJsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )

            # Custom JSON mode parser for Cloudflare
            if is_pydantic_schema:

                class CloudflarePydanticOutputParser(PydanticOutputParser):
                    def parse(self, text: str) -> Any:
                        """Parse the output of an LLM call
                        specifically for Cloudflare."""
                        if text and isinstance(text, str):
                            # Try to extract JSON from the text
                            json_str = None

                            # Check if it's already a valid JSON
                            if (text.startswith("{") and text.endswith("}")) or (
                                text.startswith("[") and text.endswith("]")
                            ):
                                json_str = text

                            # Try to parse the JSON if found
                            if json_str:
                                try:
                                    data = json.loads(json_str)
                                    # Try direct parsing
                                    try:
                                        if hasattr(
                                            self.pydantic_object, "model_validate"
                                        ):
                                            return self.pydantic_object.model_validate(
                                                data
                                            )
                                        else:
                                            return self.pydantic_object.parse_obj(data)
                                    except Exception:
                                        # If direct parsing fails,
                                        # check if data might be nested
                                        if (
                                            isinstance(data, dict)
                                            and "parameters" in data
                                        ):
                                            try:
                                                if hasattr(
                                                    self.pydantic_object,
                                                    "model_validate",
                                                ):
                                                    return (
                                                        self.pydantic_object.model_validate(
                                                            data["parameters"]
                                                        )
                                                    )
                                                else:
                                                    return (
                                                        self.pydantic_object.parse_obj(
                                                            data["parameters"]
                                                        )
                                                    )
                                            except Exception:
                                                pass
                                except json.JSONDecodeError:
                                    pass

                        # Fallback to standard parsing behavior
                        return super().parse(text)

                output_parser = CloudflarePydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
            else:

                class CloudflareJsonOutputParser(JsonOutputParser):
                    def parse(self, text: str) -> Any:
                        """Parse the output specifically for Cloudflare."""
                        if text and isinstance(text, str):
                            # Try to extract JSON from the text
                            json_str = None

                            # Check if it's already a valid JSON
                            if (text.startswith("{") and text.endswith("}")) or (
                                text.startswith("[") and text.endswith("]")
                            ):
                                json_str = text

                            # Try to parse the JSON if found
                            if json_str:
                                try:
                                    return json.loads(json_str)
                                except json.JSONDecodeError:
                                    pass

                        # Fallback to standard parsing
                        return super().parse(text)

                output_parser = CloudflareJsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
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


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


#
# Type conversion helpers
#
def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary,
    meeting Cloudflare AI Workers requirements."""
    message_dict: Dict[str, Any]

    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}

    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}

    elif isinstance(message, AIMessage):
        # For assistant messages, handle tool calls specially
        if message.tool_calls:
            # When there are tool calls, set content to None
            tool_calls = []
            for tc in message.tool_calls:
                # Convert args to a JSON string if needed
                args_str = (
                    json.dumps(tc["args"])
                    if isinstance(tc["args"], dict)
                    else tc["args"]
                )

                tool_calls.append(
                    {
                        "id": tc.get("id", str(uuid.uuid4())),  # Generate ID if missing
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": args_str},
                    }
                )

            # Create message with tool_calls but no content
            message_dict = {
                "role": "assistant",
                "content": None,  # Must be null, not empty string
                "tool_calls": tool_calls,
            }
        else:
            # For regular assistant messages without tool calls
            content = message.content
            # Important: if content is empty string, convert to null
            if content == "":
                content = None

            message_dict = {"role": "assistant", "content": content}

        # Handle legacy function_call format (backward compatibility)
        if "function_call" in message.additional_kwargs and not message.tool_calls:
            function_call = message.additional_kwargs["function_call"]
            message_dict["function_call"] = function_call
            # If function_call present, follow same pattern - set content to null
            message_dict["content"] = None

    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}

    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "name": message.name,
            "content": message.content,
        }

    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "name": message.name,
            "content": message.content,
        }

        # Only include tool_call_id if it's set
        if message.tool_call_id:
            message_dict["tool_call_id"] = message.tool_call_id

    else:
        raise TypeError(f"Got unknown type {message}")

    # Add any additional kwargs not already handled
    for key, value in message.additional_kwargs.items():
        if key not in message_dict and key not in ["function_call", "tool_calls"]:
            message_dict[key] = value

    return message_dict


def _lc_tool_call_to_cf_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_cf_tool_call(
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
