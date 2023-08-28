"""Wrapper around Fireworks APIs"""
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import requests

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class BaseFireworks(BaseLLM):
    """Wrapper around Fireworks large language models."""

    model_id: str = Field("accounts/fireworks/models/llama-v2-7b-chat", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    fireworks_api_key: Optional[str] = None
    """Api key to use fireworks API"""
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to Fireworks completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"fireworks_api_key": "FIREWORKS_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    def __new__(cls, **data: Any) -> Any:
        """Initialize the Fireworks object."""
        data.get("model_id", "")
        return super().__new__(cls)

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["fireworks_api_key"] = get_from_dict_or_env(
            values, "fireworks_api_key", "FIREWORKS_API_KEY"
        )
        return values

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Fireworks endpoint with k unique prompts.
        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The full LLM output.
        """
        params = {"model": self.model_id}
        params = {**params, **kwargs}
        sub_prompts = self.get_batch_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for _prompts in sub_prompts:
            response = completion_with_retry(self, prompt=prompts, **params)
            choices.extend(response)
            update_token_usage(_keys, response, token_usage)

        return self.create_llm_result(choices, prompts, token_usage)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Fireworks endpoint async with k unique prompts."""
        params = {"model": self.model_id}
        params = {**params, **kwargs}
        sub_prompts = self.get_batch_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for _prompts in sub_prompts:
            response = await acompletion_with_retry(self, prompt=_prompts, **params)
            choices.extend(response)
            update_token_usage(_keys, response, token_usage)

        return self.create_llm_result(choices, prompts, token_usage)

    def get_batch_prompts(
        self,
        params: Dict[str, Any],
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Get the sub prompts for llm call."""
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")

        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        return sub_prompts

    def create_llm_result(
        self, choices: Any, prompts: List[str], token_usage: Dict[str, int]
    ) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []

        for i, _ in enumerate(prompts):
            sub_choices = choices[i : (i + 1)]
            generations.append(
                [
                    Generation(
                        text=choice,
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_id": self.model_id}
        return LLMResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks"


class FireworksChat(BaseLLM):
    """Wrapper around Fireworks Chat large language models.
    To use, you should have the ``fireworksai`` python package installed, and the
    environment variable ``FIREWORKS_API_KEY`` set with your API key.
    Any parameters that are valid to be passed to the fireworks.create
    call can be passed in, even if not explicitly saved on this class.
    Example:
        .. code-block:: python
            from langchain.llms import FireworksChat
            fireworkschat = FireworksChat(model_id=""llama-v2-13b-chat"")
    """

    model_id: str = "accounts/fireworks/models/llama-v2-7b-chat"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    fireworks_api_key: Optional[str] = None
    max_retries: int = 6
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to Fireworks completion API. Default is 600 seconds."""
    """Maximum number of retries to make when generating."""
    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment"""
        values["fireworks_api_key"] = get_from_dict_or_env(
            values, "fireworks_api_key", "FIREWORKS_API_KEY"
        )
        return values

    def _get_chat_params(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> Tuple:
        if len(prompts) > 1:
            raise ValueError(
                f"FireworksChat currently only supports single prompt, got {prompts}"
            )
        messages = self.prefix_messages + [{"role": "user", "content": prompts[0]}]
        params: Dict[str, Any] = {**{"model": self.model_id}}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")

        return messages, params

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}
        full_response = completion_with_retry(self, messages=messages, **params)
        llm_output = {
            "model_id": self.model_id,
        }
        return LLMResult(
            generations=[[Generation(text=full_response[0])]],
            llm_output=llm_output,
        )

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}
        full_response = await acompletion_with_retry(self, messages=messages, **params)
        llm_output = {
            "model_id": self.model_id,
        }
        return LLMResult(
            generations=[[Generation(text=full_response[0])]],
            llm_output=llm_output,
        )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks-chat"


class Fireworks(BaseFireworks):
    """Wrapper around Fireworks large language models.
    To use, you should have the ``fireworks`` python package installed, and the
    environment variable ``FIREWORKS_API_KEY`` set with your API key.
    Any parameters that are valid to be passed to the fireworks.create
    call can be passed in, even if not explicitly saved on this class.
    Example:
        .. code-block:: python
            from langchain.llms import fireworks
            llm = Fireworks(model_id="llama-v2-13b")
    """


def update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response)
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def execute(
    prompt: str,
    model: str,
    api_key: Optional[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Any:
    """Execute LLM query"""
    requestUrl = "https://api.fireworks.ai/inference/v1/completions"
    requestBody = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    requestHeaders = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.post(requestUrl, headers=requestHeaders, json=requestBody)
    return response.text


def completion_with_retry(
    llm: Union[BaseFireworks, FireworksChat], **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    if "prompt" not in kwargs.keys():
        answers = []
        for i in range(len(kwargs["messages"])):
            result = kwargs["messages"][i]["content"]
            result = execute(
                result,
                kwargs["model"],
                llm.fireworks_api_key,
                llm.max_tokens,
                llm.temperature,
                llm.top_p,
            )
            curr_string = json.loads(result)["choices"][0]["text"]
            answers.append(curr_string)
    else:
        answers = []
        for i in range(len(kwargs["prompt"])):
            result = kwargs["prompt"][i]
            result = execute(
                result,
                kwargs["model"],
                llm.fireworks_api_key,
                llm.max_tokens,
                llm.temperature,
                llm.top_p,
            )
            curr_string = json.loads(result)["choices"][0]["text"]
            answers.append(curr_string)
    return answers


async def acompletion_with_retry(
    llm: Union[BaseFireworks, FireworksChat], **kwargs: Any
) -> Any:
    """Use tenacity to retry the async completion call."""
    if "prompt" not in kwargs.keys():
        answers = []
        for i in range(len(kwargs["messages"])):
            result = kwargs["messages"][i]["content"]
            result = execute(
                result,
                kwargs["model"],
                llm.fireworks_api_key,
                llm.max_tokens,
                llm.temperature,
            )
            curr_string = json.loads(result)["choices"][0]["text"]
            answers.append(curr_string)
    else:
        answers = []
        for i in range(len(kwargs["prompt"])):
            result = kwargs["prompt"][i]
            result = execute(
                result,
                kwargs["model"],
                llm.fireworks_api_key,
                llm.max_tokens,
                llm.temperature,
            )
            curr_string = json.loads(result)["choices"][0]["text"]
            answers.append(curr_string)
    return answers
