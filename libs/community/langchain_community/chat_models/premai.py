"""Wrapper around Prem's Chat API."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

if TYPE_CHECKING:
    from premai.api.chat_completions.v1_chat_completions_create import (
        ChatCompletionResponseStream,
    )
    from premai.models.chat_completion_response import ChatCompletionResponse

logger = logging.getLogger(__name__)


class ChatPremAPIError(Exception):
    """Error with the `PremAI` API."""


def _truncate_at_stop_tokens(
    text: str,
    stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text


def _response_to_result(
    response: ChatCompletionResponse,
    stop: Optional[List[str]],
) -> ChatResult:
    """Converts a Prem API response into a LangChain result"""

    if not response.choices:
        raise ChatPremAPIError("ChatResponse must have at least one candidate")
    generations: List[ChatGeneration] = []
    for choice in response.choices:
        role = choice.message.role
        if role is None:
            raise ChatPremAPIError(f"ChatResponse {choice} must have a role.")

        # If content is None then it will be replaced by ""
        content = _truncate_at_stop_tokens(text=choice.message.content or "", stop=stop)
        if content is None:
            raise ChatPremAPIError(f"ChatResponse must have a content: {content}")

        if role == "assistant":
            generations.append(
                ChatGeneration(text=content, message=AIMessage(content=content))
            )
        elif role == "user":
            generations.append(
                ChatGeneration(text=content, message=HumanMessage(content=content))
            )
        else:
            generations.append(
                ChatGeneration(
                    text=content, message=ChatMessage(role=role, content=content)
                )
            )
    return ChatResult(generations=generations)


def _convert_delta_response_to_message_chunk(
    response: ChatCompletionResponseStream, default_class: Type[BaseMessageChunk]
) -> Tuple[
    Union[BaseMessageChunk, HumanMessageChunk, AIMessageChunk, SystemMessageChunk],
    Optional[str],
]:
    """Converts delta response to message chunk"""
    _delta = response.choices[0].delta  # type: ignore
    role = _delta.get("role", "")  # type: ignore
    content = _delta.get("content", "")  # type: ignore
    additional_kwargs: Dict = {}

    if role is None or role == "":
        raise ChatPremAPIError("Role can not be None. Please check the response")

    finish_reasons: Optional[str] = response.choices[0].finish_reason

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content), finish_reasons
    elif role == "assistant" or default_class == AIMessageChunk:
        return (
            AIMessageChunk(content=content, additional_kwargs=additional_kwargs),
            finish_reasons,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content), finish_reasons
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role), finish_reasons
    else:
        return default_class(content=content), finish_reasons


def _messages_to_prompt_dict(
    input_messages: List[BaseMessage],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Converts a list of LangChain Messages into a simple dict
    which is the message structure in Prem"""

    system_prompt: Optional[str] = None
    examples_and_messages: List[Dict[str, str]] = []

    for input_msg in input_messages:
        if isinstance(input_msg, SystemMessage):
            system_prompt = str(input_msg.content)
        elif isinstance(input_msg, HumanMessage):
            examples_and_messages.append(
                {"role": "user", "content": str(input_msg.content)}
            )
        elif isinstance(input_msg, AIMessage):
            examples_and_messages.append(
                {"role": "assistant", "content": str(input_msg.content)}
            )
        else:
            raise ChatPremAPIError("No such role explicitly exists")
    return system_prompt, examples_and_messages


class ChatPremAI(BaseChatModel, BaseModel):
    """Use any LLM provider with Prem and Langchain.

    To use, you will need to have an API key. You can find your existing API Key
    or generate a new one here: https://app.premai.io/api_keys/
    """

    # TODO: Need to add the default parameters through prem-sdk here

    project_id: int
    """The project ID in which the experiments or deployments are carried out. 
    You can find all your projects here: https://app.premai.io/projects/"""
    premai_api_key: Optional[SecretStr] = None
    """Prem AI API Key. Get it here: https://app.premai.io/api_keys/"""

    model: Optional[str] = None
    """Name of the model. This is an optional parameter. 
    The default model is the one deployed from Prem's LaunchPad: https://app.premai.io/projects/8/launchpad
    If model name is other than default model then it will override the calls 
    from the model deployed from launchpad."""

    session_id: Optional[str] = None
    """The ID of the session to use. It helps to track the chat history."""

    temperature: Optional[float] = None
    """Model temperature. Value should be >= 0 and <= 1.0"""

    top_p: Optional[float] = None
    """top_p adjusts the number of choices for each predicted tokens based on
        cumulative probabilities. Value should be ranging between 0.0 and 1.0. 
    """

    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate"""

    max_retries: int = 1
    """Max number of retries to call the API"""

    system_prompt: Optional[str] = ""
    """Acts like a default instruction that helps the LLM act or generate 
    in a specific way.This is an Optional Parameter. By default the 
    system prompt would be using Prem's Launchpad models system prompt. 
    Changing the system prompt would override the default system prompt.
    """

    streaming: Optional[bool] = False
    """Whether to stream the responses or not."""

    tools: Optional[Dict[str, Any]] = None
    """A list of tools the model may call. Currently, only functions are 
    supported as a tool"""

    frequency_penalty: Optional[float] = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based"""

    presence_penalty: Optional[float] = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based 
    on whether they appear in the text so far."""

    logit_bias: Optional[dict] = None
    """JSON object that maps tokens to an associated bias value from -100 to 100."""

    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens."""

    seed: Optional[int] = None
    """This feature is in Beta. If specified, our system will make a best effort 
    to sample deterministically."""

    client: Any

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environments(cls, values: Dict) -> Dict:
        """Validate that the package is installed and that the API token is valid"""
        try:
            from premai import Prem
        except ImportError as error:
            raise ImportError(
                "Could not import Prem Python package."
                "Please install it with: `pip install premai`"
            ) from error

        try:
            premai_api_key = get_from_dict_or_env(
                values, "premai_api_key", "PREMAI_API_KEY"
            )
            values["client"] = Prem(api_key=premai_api_key)
        except Exception as error:
            raise ValueError("Your API Key is incorrect. Please try again.") from error
        return values

    @property
    def _llm_type(self) -> str:
        return "premai"

    @property
    def _default_params(self) -> Dict[str, Any]:
        # FIXME: n and stop is not supported, so hardcoding to current default value
        return {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "logit_bias": self.logit_bias,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "stop": None,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        all_kwargs = {**self._default_params, **kwargs}
        for key in list(self._default_params.keys()):
            if all_kwargs.get(key) is None or all_kwargs.get(key) == "":
                all_kwargs.pop(key, None)
        return all_kwargs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, messages_to_pass = _messages_to_prompt_dict(messages)  # type: ignore

        kwargs["stop"] = stop
        if system_prompt is not None and system_prompt != "":
            kwargs["system_prompt"] = system_prompt

        all_kwargs = self._get_all_kwargs(**kwargs)
        response = chat_with_retry(
            self,
            project_id=self.project_id,
            messages=messages_to_pass,
            stream=False,
            run_manager=run_manager,
            **all_kwargs,
        )

        return _response_to_result(response=response, stop=stop)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        system_prompt, messages_to_pass = _messages_to_prompt_dict(messages)
        kwargs["stop"] = stop

        if "system_prompt" not in kwargs:
            if system_prompt is not None and system_prompt != "":
                kwargs["system_prompt"] = system_prompt

        all_kwargs = self._get_all_kwargs(**kwargs)

        default_chunk_class = AIMessageChunk

        for streamed_response in chat_with_retry(
            self,
            project_id=self.project_id,
            messages=messages_to_pass,
            stream=True,
            run_manager=run_manager,
            **all_kwargs,
        ):
            try:
                chunk, finish_reason = _convert_delta_response_to_message_chunk(
                    response=streamed_response, default_class=default_chunk_class
                )
                generation_info = (
                    dict(finish_reason=finish_reason)
                    if finish_reason is not None
                    else None
                )
                cg_chunk = ChatGenerationChunk(
                    message=chunk, generation_info=generation_info
                )
                if run_manager:
                    run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
                yield cg_chunk
            except Exception as _:
                continue


def create_prem_retry_decorator(
    llm: ChatPremAI,
    *,
    max_retries: int = 1,
    run_manager: Optional[Union[CallbackManagerForLLMRun]] = None,
) -> Callable[[Any], Any]:
    import premai.models

    errors = [
        premai.models.api_response_validation_error.APIResponseValidationError,
        premai.models.conflict_error.ConflictError,
        premai.models.model_not_found_error.ModelNotFoundError,
        premai.models.permission_denied_error.PermissionDeniedError,
        premai.models.provider_api_connection_error.ProviderAPIConnectionError,
        premai.models.provider_api_status_error.ProviderAPIStatusError,
        premai.models.provider_api_timeout_error.ProviderAPITimeoutError,
        premai.models.provider_internal_server_error.ProviderInternalServerError,
        premai.models.provider_not_found_error.ProviderNotFoundError,
        premai.models.rate_limit_error.RateLimitError,
        premai.models.unprocessable_entity_error.UnprocessableEntityError,
        premai.models.validation_error.ValidationError,
    ]

    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=run_manager
    )
    return decorator


def chat_with_retry(
    llm: ChatPremAI,
    project_id: int,
    messages: List[dict],
    stream: bool = False,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Using tenacity for retry in completion call"""
    retry_decorator = create_prem_retry_decorator(
        llm, max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _completion_with_retry(
        project_id: int,
        messages: List[dict],
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> Any:
        response = llm.client.chat.completions.create(
            project_id=project_id,
            messages=messages,
            stream=stream,
            **kwargs,
        )
        return response

    return _completion_with_retry(
        project_id=project_id,
        messages=messages,
        stream=stream,
        **kwargs,
    )
