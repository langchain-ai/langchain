"""Wrapper around Prem AI Platform"""

import logging
from typing import Any, Dict, List, Union, Optional, Callable

from langchain_core.language_models.llms import BaseLLM
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.language_models import LanguageModelInput

logger = logging.getLogger(__name__)


class Prem(BaseLLM, BaseModel):
    """Use any LLM provider with Prem and Langchain.

    To use, you will need to have an API key. You can find your existing API Key
    or generate a new one here: https://app.premai.io/api_keys/
    """

    model: str
    premai_api_key: str
    """Prem AI API Key. Get it here: https://app.premai.io/api_keys/"""

    project_id: int
    """The project ID in which the experiments or deployements are carried out. You can find all your projects here: https://app.premai.io/projects/"""

    session_id: Optional[str] = None
    """The ID of the session to use. It helps to track the chat history."""

    temperature: Optional[float] = None
    """Model temperature. Value shoud be >= 0 and <= 1.0"""

    top_p: Optional[float] = None
    """top_p adjusts the number of choices for each predicted tokens based on
        cumulative probabilities. Value should be ranging between 0.0 and 1.0. 
    """

    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate"""

    max_retries: Optional[int] = 1
    """Max number of retries to call the API"""

    system_prompt: Optional[str] = ""
    """Acts like a default instruction that helps the LLM act or generate in a specific way."""

    n: Optional[int] = None
    """The number of responses to generate."""

    streaming: Optional[bool] = False
    """Whether to stream the responses or not."""

    tools: Optional[Dict[str, Any]] = None
    """A list of tools the model may call. Currently, only functions are supported as a tool"""

    frequency_penalty: Optional[float] = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based"""

    presence_penalty: Optional[float] = None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far."""

    logit_bias: Optional[dict] = None
    """JSON object that maps tokens to an associated bias value from -100 to 100."""

    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens."""

    seed: Optional[int] = None
    """This feature is in Beta. If specified, our system will make a best effort to sample deterministically."""

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
        return "prem"

    @property
    def _default_params(self) -> Dict[str, Any]:
        # For default objects tools can not be None
        return {
            "model": "gpt-3.5-turbo",
            "system_prompt": "",
            "top_p": 0.95,
            "temperature": 1.0,
            "stream": False,
            "n": 1,
            "logit_bias": None,
            "max_tokens": 128,
            "presence_penalty": 0.0,
            "frequency_penalty": 2,
            "seed": None,
            "stop": None,
        }

    def _apply_default_format(self, prompt: str) -> Dict[str, str]:
        if self.system_prompt != "":
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            return [
                {"role": "user", "content": prompt},
            ]

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:

        messages = [self._apply_default_format(prompt) for prompt in prompts]
        generations: List[List[Generation]] = []

        if stop is not None:
            kwargs["stop"] = stop

        responses = completion_with_retry(
            self,
            project_id=self.project_id,
            prompt=messages,
            stream=False,
            run_manager=run_manager,
            **kwargs,
        )
        prompt_generation = []
        for response in responses.choices:
            prompt_generation.append(response.message.content)
            generations.append(prompt_generation)
        print(generations)
        return LLMResult(generations=generations)


def create_prem_retry_decorator(
    llm: Prem,
    *,
    max_retries: int = 1,
    run_manager: Optional[
        Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]
    ] = None,
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


def completion_with_retry(
    llm: Prem,
    project_id: int,
    prompt: LanguageModelInput,
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
        project_id: int, prompt: LanguageModelInput, stream: bool, **kwargs: Any
    ) -> Any:
        print("================================")
        print(prompt[0])
        print(project_id)
        return llm.client.chat.completions.create(
            project_id=project_id, messages=prompt[0], stream=stream, **kwargs
        )

    return _completion_with_retry(
        project_id=project_id, prompt=prompt, stream=stream, **kwargs
    )
