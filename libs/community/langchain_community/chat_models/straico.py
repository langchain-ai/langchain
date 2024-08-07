import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_community.utilities.requests import Requests


class ChatStraico(BaseChatModel):
    """
    Straico is a platform that provides access to a variety of popular LLMs
    for text, images and audio generation under one coin-based pricing system.
    Straico's API supports text completion.

    More information on the API:
    https://documenter.getpostman.com/view/5900072/2s9YyzddrR

    To use this chat model, set your API key as `STRAICO_API_KEY`
    or pass it as `api_key`.
    """

    model: Optional[str] = Field(default="openai/gpt-3.5-turbo-0125")
    """Model name to use."""
    straico_api_key: Optional[SecretStr] = Field(None, alias="api_key")
    """Automatically inferred from env var `STRAICO_API_KEY` if not provided."""
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Straico completion API. Can be float, httpx.Timeout or 
        None."""
    max_retries: int = Field(default=2)
    """Maximum number of retries to make when generating."""
    # max_tokens: Optional[int] = None
    # """Maximum number of tokens to generate."""

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["straico_api_key"] = get_from_dict_or_env(
            values, "straico_api_key", "STRAICO_API_KEY"
        )
        return values

    def _generate(
        self,
        messages,
        stop=None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params}
        response = self._completion_with_retry(
            messages=message_dicts, run_manager=run_manager, params=params
        )
        return self._create_chat_result(response["data"]["completion"])

    async def _agenerate(self, messages, stop=None, run_manager=None) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params}
        response = await self._acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, params=params
        )
        return self._create_chat_result(response["data"]["completion"])

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response["usage"]
        llm_output = {"token_usage": token_usage, "model": self.model}
        res = ChatResult(generations=generations, llm_output=llm_output)
        return res

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Straico API."""
        return {
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            # "max_tokens": self.max_tokens,
        }

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        straico_creds: Dict[str, Any] = {
            "api_key": self.straico_api_key,
            "url": "https://api.straico.com/v0/prompt/completion",
            "model": self.model,
        }
        return {**straico_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "straicochat"

    def _completion_with_retry(
        self,
        messages: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        params: Dict[str, Any] = {},
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(
            messages: List[Dict[str, Any]], params: Dict[str, Any]
        ) -> Any:
            auth_header = {"Authorization": f"Bearer {params['api_key']}"}
            try:
                request_timeout = params.get("request_timeout", None)
                request = Requests(headers=auth_header)
                # Create a payload dictionary
                payloadStruct = {
                    "model": params["model"],
                    "message": str(messages),
                }
                response = request.post(
                    url=params["url"], data=payloadStruct, timeout=request_timeout
                )
                self._handle_status(response.status_code, response.text)
                response_data = json.loads(response._content)
                return response_data
            except Exception as e:
                raise e

        return _completion_with_retry(messages=messages, params=params)

    async def _acompletion_with_retry(
        self,
        messages: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        params: Dict[str, Any] = {},
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(
            messages: List[Dict[str, Any]], params: Dict[str, Any]
        ) -> Any:
            auth_header = {"Authorization": f"Bearer {params['api_key']}"}
            try:
                request = Requests(headers=auth_header)
                # Create a payload dictionary
                payloadStruct = {"model": params["model"], "message": str(messages)}
                response = request.post(url=params["url"], data=payloadStruct)
                self._handle_status(response.status_code, response.text)
                response_data = json.loads(response._content)
                return response_data
            except Exception as e:
                raise e

        return await _completion_with_retry(messages=messages, params=params)

    def _handle_status(self, status_code: int, text: str) -> None:
        if status_code >= 500:
            raise Exception(f"Straico Server: Error {status_code}")
        elif status_code >= 400:
            raise ValueError(f"Straico received an invalid payload: {text}")
        elif status_code != 201:  # straico returns 201 for success
            raise Exception(f"Request failed with status code {status_code}: {text}")

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        """Custom method to combine the llm_output information for batched call."""
        overall_token_usage: dict = {}
        for output in llm_outputs:
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
        combined = {"token_usage": overall_token_usage, "model_name": self.model}
        return combined


def _create_retry_decorator(
    llm: ChatStraico,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    return create_base_retry_decorator(
        error_types=[requests.exceptions.ConnectTimeout],
        max_retries=llm.max_retries,
        run_manager=run_manager,
    )
