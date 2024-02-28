"""Wrapper around Prem AI Platform"""

from typing import Any, Dict, List, Union, Optional, Callable

from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, SecretStr
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator


class Prem(LLM):
    """Use any LLM provider with Prem and Langchain.

    To use, you will need to have an API key. You can find your existing API Key
    or generate a new one here: https://app.premai.io/api_keys/
    """

    model: str
    premai_api_key: SecretStr
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

    system_prompt: Optional[str] = ""
    """Acts like a default instruction that helps the LLM act or generate in a specific way."""

    n: Optional[int] = None
    """The number of responses to generate."""

    # stream: Optional[bool] = False
    # """Whether to stream the responses or not."""

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

    @property
    def _llm_type(self) -> str:
        return "prem"

    @property
    def default_params(self) -> Dict[str, Any]:
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
            "frequency_penalty": 0.0,
            "seed": None,
            "tools": None,
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
        prompts: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        pass

    def _call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Calls out models supported in Prem.

        Args:
            prompt: The prompt to pass into the model
        """

        try:
            from premai import Prem
        except ImportError as error:
            raise ImportError(
                "Could not import Prem Python package."
                "Please install it with: `pip install premai`"
            ) from error

        try:
            client = Prem(api_key=str(self.premai_api_key.get_secret_value()))
        except ValueError as error:
            raise ValueError("Your API Key is incorrect. Please try again.") from error

        prompt = (
            self._apply_default_format(prompt) if isinstance(prompt, str) else prompt
        )
        response = client.chat.completions.create(
            project_id=self.project_id, messages=prompt, **kwargs
        )
        return response


def completion_with_reply(
    llm: Prem,
    use_retry: bool,
    *,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Using tenacity to retry the completion call"""


def _retry_decorator(
    llm: Prem,
    *,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Define retry mechanism."""
    # TODO: Add custom Prem Errors
    errors = []
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
