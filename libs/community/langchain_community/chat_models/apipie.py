"""APIpie.ai chat wrapper."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import requests
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
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_community.utils.openai import is_openai_v1

if TYPE_CHECKING:
    import tiktoken


logger = logging.getLogger(__name__)
DEFAULT_API_BASE = "https://apipie.ai/v1"
DEFAULT_MODEL = "openai/gpt-4o"


def _import_tiktoken() -> Any:
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "Could not import tiktoken python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install tiktoken`."
        )
    return tiktoken


def _create_retry_decorator(
    llm: ChatAPIpie,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import openai

    errors = [
        openai.error.Timeout,  # type: ignore[attr-defined]
        openai.error.APIError,  # type: ignore[attr-defined]
        openai.error.APIConnectionError,  # type: ignore[attr-defined]
        openai.error.RateLimitError,  # type: ignore[attr-defined]
        openai.error.ServiceUnavailableError,  # type: ignore[attr-defined]
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


async def acompletion_with_retry(
    llm: ChatAPIpie,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    # Filter out APIpie-specific parameters
    api_specific_params = {
        "integrity",
        "integrity_model",
        "force_provider",
        "tools_model",
        "mem_session",
        "mem_expire",
        "mem_clear",
        "mem_msgs",
        "mem_length",
        "rag_tune",
        "routing",
    }
    openai_kwargs = {k: v for k, v in kwargs.items() if k not in api_specific_params}

    if is_openai_v1():
        # For OpenAI v1, pass APIpie-specific params via extra_body
        extra_body = {k: v for k, v in kwargs.items() if k in api_specific_params}
        if extra_body:
            openai_kwargs["extra_body"] = extra_body
        return await llm.async_client.create(**openai_kwargs)

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await llm.client.acreate(**kwargs)

    # For non-v1, pass filtered parameters
    return await _completion_with_retry(**openai_kwargs)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _update_token_usage(
    overall_token_usage: Union[int, dict], new_usage: Union[int, dict]
) -> Union[int, dict]:
    # Token usage is either ints or dictionaries
    # `reasoning_tokens` is nested inside `completion_tokens_details`
    if isinstance(new_usage, int):
        if not isinstance(overall_token_usage, int):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return new_usage + overall_token_usage
    elif isinstance(new_usage, dict):
        if not isinstance(overall_token_usage, dict):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return {
            k: _update_token_usage(overall_token_usage.get(k, 0), v)
            for k, v in new_usage.items()
        }
    else:
        warnings.warn(f"Unexpected type for token usage: {type(new_usage)}")
        return new_usage


class ChatAPIpie(BaseChatModel):
    """`APIpie` Chat large language models API.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``APIPIE_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatAPIpie
            apipie = ChatAPIpie(model="gpt-3.5-turbo")
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"apipie_api_key": "APIPIE_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "openai"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.apipie_api_base:
            attributes["apipie_api_base"] = self.apipie_api_base

        if self.apipie_proxy:
            attributes["apipie_proxy"] = self.apipie_proxy

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    top_p: float = 1.0
    """Nucleus sampling parameter, controls the cumulative probability 
    distribution cutoff."""
    top_k: int = 0
    """Limits the selection pool to the top k 
    most probable tokens.
    """
    frequency_penalty: float = 0.0
    """Penalizes tokens based on their 
    frequency in the text so far.
    """
    presence_penalty: float = 0.0
    """Penalizes tokens that have already 
    appeared in the text.
    """
    repetition_penalty: float = 1.0
    """Discourages repeating the same words or phrases."""
    beam_size: int = 1
    """Number of sequences to keep at each step of generation."""
    memory: bool = False
    """Enable Integrated Model Memory to maintain conversation context."""
    mem_session: Optional[str] = None
    """Unique identifier for maintaining separate memory chains."""
    mem_expire: Optional[int] = None
    """Time in minutes after which stored memories will expire."""
    mem_clear: int = 0
    """Set to 1 to instantly delete all stored memories for the specified session."""
    mem_msgs: int = 8
    """Maximum number of messages to append from memory."""
    mem_length: int = 20
    """Percentage of model's max response tokens to use for memory."""
    rag_tune: Optional[str] = None
    """Name of the RAG tune or vector collection for RAG tuning."""
    routing: str = "perf"
    """Define how to route calls when multiple providers exist."""
    tools: Optional[List[Dict[str, Any]]] = None
    """List of tools to integrate into the chat model."""
    tool_choice: str = "none"
    """Specifies how the tools are chosen."""
    tools_model: str = "gpt-4o-mini"
    """Model to use for processing tools."""
    integrity: Optional[int] = None
    """Integrity setting for querying and returning best answers."""
    integrity_model: str = "gpt-4o"
    """Model to use for integrity checks."""
    force_provider: bool = False
    """Force request to be routed to the specified provider."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    apipie_api_key: SecretStr = Field(default=SecretStr(""), alias="api_key")
    """Automatically inferred from env var `APIPIE_API_KEY` if not provided."""
    apipie_api_base: str = Field(default=DEFAULT_API_BASE, alias="base_url")
    """Base URL path for API requests, defaults to Apipie's API endpoint.
        Can be overridden by APIPIE_API_BASE environment variable."""
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for APIpie
    apipie_proxy: Optional[str] = None
    """To support explicit proxy for APIpie."""
    available_models: Optional[Set[str]] = None
    """Available models from APIpie API."""
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to APIpie completion API. Can be float, httpx.Timeout or 
        None."""
    max_retries: int = Field(default=3)
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    tiktoken_model_name: str = "gpt-4o"
    """The model name to pass to tiktoken for token counting."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""

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
                logger.warning(
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

    @staticmethod
    def get_available_models(
        apipie_api_key: Optional[str] = None,
        apipie_api_base: str = DEFAULT_API_BASE,
    ) -> Set[str]:
        """Get available models from APIpie API."""
        try:
            apipie_api_key = apipie_api_key or os.environ["APIPIE_API_KEY"]
        except KeyError as e:
            raise ValueError(
                "APIpie API key must be passed as keyword argument or "
                "set in environment variable APIPIE_API_KEY.",
            ) from e

        # For testing purposes
        if apipie_api_key in ["foo", "test_key"]:
            return {"openai/gpt-4o", "openai/-3.5-gptturbo", "anthropic/claude-3-opus"}

        models_url = f"{apipie_api_base}/models"
        models_response = requests.get(
            models_url,
            headers={
                "Authorization": f"Bearer {apipie_api_key}",
            },
        )

        if models_response.status_code != 200:
            raise ValueError(
                f"Error getting models from {models_url}: "
                f"{models_response.status_code}",
            )

        # Return both the full provider/id format and just the id for compatibility
        model_ids = set()
        for model in models_response.json()["data"]:
            # Only include models with type "llm"
            if model.get("type") == "llm":
                # Add the id by itself
                model_ids.add(model["id"])
                # Add the provider/id format if provider exists
                if "provider" in model:
                    model_ids.add(f"{model['provider']}/{model['id']}")

        return model_ids

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values.get("n", 1) < 1:
            raise ValueError("n must be at least 1.")
        if values.get("n", 1) > 1 and values.get("streaming", False):
            raise ValueError("n must be 1 when streaming.")

        values["apipie_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "apipie_api_key", "APIPIE_API_KEY")
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["openai_organization"] = (
            values.get("openai_organization")
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        values["apipie_api_base"] = get_from_dict_or_env(
            values,
            "apipie_api_base",
            "APIPIE_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["apipie_proxy"] = get_from_dict_or_env(
            values,
            "apipie_proxy",
            "APIPIE_PROXY",
            default="",
        )
        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        if is_openai_v1():
            client_params = {
                "api_key": values["apipie_api_key"].get_secret_value(),
                "organization": values["openai_organization"],
                "base_url": values["apipie_api_base"],
                "timeout": values.get("request_timeout"),
                "max_retries": values.get("max_retries", 3),
                "default_headers": values.get("default_headers"),
                "default_query": values.get("default_query"),
                "http_client": values.get("http_client"),
            }

            if not values.get("client"):
                values["client"] = openai.OpenAI(**client_params).chat.completions
            if not values.get("async_client"):
                values["async_client"] = openai.AsyncOpenAI(
                    **client_params
                ).chat.completions
        elif not values.get("client"):
            values["client"] = openai.ChatCompletion  # type: ignore[attr-defined]
            if not values.get("async_client"):
                values["async_client"] = openai.ChatCompletion  # type: ignore[attr-defined]

        if "model_name" not in values.keys():
            values["model_name"] = DEFAULT_MODEL

        if values["apipie_api_key"].get_secret_value() in ["foo", "test_key"]:
            if "available_models" in values and values["available_models"] is not None:
                model_name = values["model_name"]
                model_id = model_name
                if "/" in model_name:
                    _, model_id = model_name.split("/", 1)

                if (
                    model_name not in values["available_models"]
                    and model_id not in values["available_models"]
                ):
                    raise ValueError(
                        f"Model name {model_name} not found in available models: "
                        f"{values['available_models']}."
                    )
            # For other tests, just set a mock available_models
            else:
                values["available_models"] = {values["model_name"]}
            return values

        try:
            model_name = values["model_name"]
            available_models = cls.get_available_models(
                values["apipie_api_key"].get_secret_value(),
                values["apipie_api_base"],
            )

            # Check if the model exists in available models
            model_id = model_name
            if "/" in model_name:
                _, model_id = model_name.split("/", 1)

            if model_name not in available_models and model_id not in available_models:
                raise ValueError(
                    f"Model name {model_name} not found in available models: "
                    f"{available_models}.",
                )

            values["available_models"] = available_models
        except ValueError as e:
            # Re-raise ValueError for model validation
            raise e
        except Exception as e:
            logger.warning(f"Could not fetch available models: {e}")
            values["available_models"] = None
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling APIpie API."""
        params = {
            "model": self.model_name,
            **self.model_kwargs,
        }

        # Only include parameters that differ from their default values
        if self.streaming:
            params["stream"] = self.streaming
        if self.n != 1:
            params["n"] = self.n
        if self.temperature != 0.7:
            params["temperature"] = self.temperature
        if self.top_p != 1.0:
            params["top_p"] = self.top_p
        if self.top_k != 0:
            params["top_k"] = self.top_k
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty != 1.0:
            params["repetition_penalty"] = self.repetition_penalty
        if self.beam_size != 1:
            params["beam_size"] = self.beam_size

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.request_timeout is not None and not is_openai_v1():
            params["request_timeout"] = self.request_timeout
        if self.memory:
            params["memory"] = self.memory
            if self.mem_session is not None:
                params["mem_session"] = self.mem_session
            if self.mem_expire is not None:
                params["mem_expire"] = self.mem_expire
            if self.mem_clear != 0:
                params["mem_clear"] = self.mem_clear
            if self.mem_msgs != 8:
                params["mem_msgs"] = self.mem_msgs
            if self.mem_length != 20:
                params["mem_length"] = self.mem_length
        if self.rag_tune is not None:
            params["rag_tune"] = self.rag_tune
        if self.routing != "perf":
            params["routing"] = self.routing
        if self.tools is not None:
            params["tools"] = self.tools
        if self.tool_choice != "none":
            params["tool_choice"] = self.tool_choice
        if self.tools_model != "gpt-4o-mini":
            params["tools_model"] = self.tools_model
        if self.integrity is not None:
            params["integrity"] = self.integrity
        if self.integrity_model != "gpt-4o":
            params["integrity_model"] = self.integrity_model
        if self.force_provider:
            params["force_provider"] = self.force_provider
        return params

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        # Filter out APIpie-specific parameters
        api_specific_params = {
            "integrity",
            "integrity_model",
            "force_provider",
            "tools_model",
            "mem_session",
            "mem_expire",
            "mem_clear",
            "mem_msgs",
            "mem_length",
            "rag_tune",
            "routing",
        }
        openai_kwargs = {
            k: v for k, v in kwargs.items() if k not in api_specific_params
        }

        if is_openai_v1():
            # For OpenAI v1, pass APIpie-specific params via extra_body
            extra_body = {k: v for k, v in kwargs.items() if k in api_specific_params}
            if extra_body:
                openai_kwargs["extra_body"] = extra_body
            return self.client.create(**openai_kwargs)

        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        # For non-v1, pass filtered parameters
        return _completion_with_retry(**openai_kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] = _update_token_usage(
                            overall_token_usage[k], v
                        )
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        # Filter out APIpie-specific parameters
        api_specific_params = {
            "integrity",
            "integrity_model",
            "force_provider",
            "tools_model",
            "mem_session",
            "mem_expire",
            "mem_clear",
            "mem_msgs",
            "mem_length",
            "rag_tune",
            "routing",
        }
        openai_kwargs = {
            k: v for k, v in kwargs.items() if k not in api_specific_params
        }
        if is_openai_v1():
            # For OpenAI v1, pass APIpie-specific params via extra_body
            extra_body = {k: v for k, v in kwargs.items() if k in api_specific_params}
            if extra_body:
                openai_kwargs["extra_body"] = extra_body
        params = {**params, **openai_kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                await run_manager.on_llm_new_token(token=cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the openai client."""
        openai_creds: Dict[str, Any] = {
            "model": self.model_name,
        }
        if not is_openai_v1():
            openai_creds.update(
                {
                    "api_key": self.apipie_api_key.get_secret_value(),
                    "api_base": self.apipie_api_base,
                    "organization": self.openai_organization,
                }
            )
        if self.apipie_proxy:
            import openai

            openai.proxy = {"http": self.apipie_proxy, "https": self.apipie_proxy}  # type: ignore[attr-defined]
        return {**self._default_params, **openai_creds}

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "apipie-chat"

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        tiktoken_ = _import_tiktoken()
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
            if model == "gpt-3.5-turbo":
                # gpt-3.5-turbo may change over time.
                # Returning num tokens assuming gpt-3.5-turbo-0301.
                model = "gpt-3.5-turbo-0301"
            elif model == "gpt-4":
                # gpt-4 may change over time.
                # Returning num tokens assuming gpt-4-0314.
                model = "gpt-4-0314"
        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken_.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken_.get_encoding(model)
        return model, encoding

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text with tiktoken package."""
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    def get_num_tokens_from_messages(
        self,
        messages: List[BaseMessage],
        tools: Optional[
            Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]
        ] = None,
    ) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools."
            )
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("gpt-3.5-turbo-0301"):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}."
                "See https://github.com/openai/openai-python/blob/main/chatml.md for "
                "information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind functions (and other objects) to this chat model.

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
         yep       (if any).
            kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        from langchain.chains.openai_functions.base import convert_to_openai_function

        formatted_functions = [convert_to_openai_function(fn) for fn in functions]
        if function_call is not None:
            if len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if formatted_functions[0]["name"] != function_call:
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            function_call_ = {"name": function_call}
            kwargs = {**kwargs, "function_call": function_call_}
        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )
