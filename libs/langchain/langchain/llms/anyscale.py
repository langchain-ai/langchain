"""Wrapper around Anyscale Endpoint"""
import os
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

import requests
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.openai import (
    BaseOpenAI,
    acompletion_with_retry,
    completion_with_retry,
)
from langchain.utils import get_from_dict_or_env
from langchain.utils.openai import is_openai_v1

DEFAULT_BASE_URL = "https://api.endpoints.anyscale.com/v1"
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


def update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def create_llm_result(
    choices: Any, prompts: List[str], token_usage: Dict[str, int], model_name: str
) -> LLMResult:
    """Create the LLMResult from the choices and prompts."""
    generations = []
    for i, _ in enumerate(prompts):
        choice = choices[i]
        generations.append(
            [
                Generation(
                    text=choice["message"]["content"],
                    generation_info=dict(
                        finish_reason=choice.get("finish_reason"),
                        logprobs=choice.get("logprobs"),
                    ),
                )
            ]
        )
    llm_output = {"token_usage": token_usage, "model_name": model_name}
    return LLMResult(generations=generations, llm_output=llm_output)


class Anyscale(BaseOpenAI):
    """Anyscale large language models.

    To use, you should have the environment variable ``ANYSCALE_API_KEY``set with your
    Anyscale Endpoint, or pass it as a named parameter to the constructor.
    To use with Anyscale Private Endpoint, please also set ``ANYSCALE_BASE_URL``.

    Example:
        .. code-block:: python
            from langchain.llms import Anyscale
            anyscalellm = Anyscale(anyscale_api_key="ANYSCALE_API_KEY")
            # To leverage Ray for parallel processing
            @ray.remote(num_cpus=1)
            def send_query(llm, text):
                resp = llm(text)
                return resp
            futures = [send_query.remote(anyscalellm, text) for text in texts]
            results = ray.get(futures)
    """

    """Key word arguments to pass to the model."""
    anyscale_api_base: str = Field(default=DEFAULT_BASE_URL)
    anyscale_api_key: SecretStr
    model_name: str = Field(default=DEFAULT_MODEL)

    prefix_messages: List = Field(default_factory=list)

    @staticmethod
    def get_available_models(
        anyscale_api_key: str = "",
        anyscale_api_base: str = DEFAULT_BASE_URL,
    ) -> Set[str]:
        """Get available models from Anyscale API."""
        try:
            anyscale_api_key = anyscale_api_key or os.environ["ANYSCALE_API_KEY"]
        except KeyError as e:
            raise ValueError(
                "Anyscale API key must be passed as keyword argument or "
                "set in environment variable ANYSCALE_API_KEY.",
            ) from e

        models_url = f"{anyscale_api_base}/models"
        models_response = requests.get(
            models_url,
            headers={
                "Authorization": f"Bearer {anyscale_api_key}",
            },
        )

        if models_response.status_code != 200:
            raise ValueError(
                f"Error getting models from {models_url}: "
                f"{models_response.status_code}",
            )

        return {model["id"] for model in models_response.json()["data"]}

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["anyscale_api_base"] = get_from_dict_or_env(
            values,
            "anyscale_api_base",
            "ANYSCALE_API_BASE",
            default=DEFAULT_BASE_URL,
        )
        values["anyscale_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "anyscale_api_key", "ANYSCALE_API_KEY")
        )
        values["model_name"] = get_from_dict_or_env(
            values,
            "model_name",
            "MODEL_NAME",
            default=DEFAULT_MODEL,
        )
        model_name = values["model_name"]
        if model_name != "test":
            available_models = cls.get_available_models(
                values["anyscale_api_key"].get_secret_value(),
                values["anyscale_api_base"],
            )

            if model_name not in available_models:
                raise ValueError(
                    f"Model name {model_name} not found in available models: "
                    f"{available_models}.",
                )

        try:
            import openai

            if is_openai_v1():
                client_params = {
                    "api_key": values["anyscale_api_key"].get_secret_value(),
                    "base_url": values["anyscale_api_base"],
                    # To do: future support
                    # "organization": values["openai_organization"],
                    # "timeout": values["request_timeout"],
                    # "max_retries": values["max_retries"],
                    # "default_headers": values["default_headers"],
                    # "default_query": values["default_query"],
                    # "http_client": values["http_client"],
                }
                values["client"] = openai.OpenAI(**client_params).chat.completions
            else:
                values["client"] = openai.ChatCompletion
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_name": self.model_name},
            **super()._identifying_params,
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        openai_creds: Dict[str, Any] = {
            "model": self.model_name,
        }
        if not is_openai_v1():
            openai_creds.update(
                {
                    "api_key": self.anyscale_api_key.get_secret_value(),
                    "api_base": self.anyscale_api_base,
                }
            )
        return {**openai_creds, **super()._invocation_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Anyscale LLM"

    def _get_chat_messages(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> Tuple:
        if len(prompts) > 1:
            raise ValueError(
                f"Anyscale currently only supports single prompt, got {prompts}"
            )
        messages = self.prefix_messages + [{"role": "user", "content": prompts[0]}]
        params: Dict[str, Any] = self._invocation_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params.get("max_tokens") == -1:
            # for Chat api, omitting max_tokens is equivalent to having no limit
            del params["max_tokens"]
        return messages, params

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        messages, params = self._get_chat_messages([prompt], stop)
        params = {**params, **kwargs, "stream": True}
        for stream_resp in completion_with_retry(
            self, messages=messages, run_manager=run_manager, **params
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            token = stream_resp["choices"][0]["delta"].get("content", "")
            if token:
                chunk = GenerationChunk(text=token)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        messages, params = self._get_chat_messages([prompt], stop)
        params = {**params, **kwargs, "stream": True}
        async for stream_resp in await acompletion_with_retry(
            self, messages=messages, run_manager=run_manager, **params
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            token = stream_resp["choices"][0]["delta"].get("content", "")
            if token:
                chunk = GenerationChunk(text=token)
                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(token, chunk=chunk)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        choices = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for prompt in prompts:
            if self.streaming:
                generation: Optional[GenerationChunk] = None
                for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    {
                        "message": {"content": generation.text},
                        "finish_reason": generation.generation_info.get("finish_reason")
                        if generation.generation_info
                        else None,
                        "logprobs": generation.generation_info.get("logprobs")
                        if generation.generation_info
                        else None,
                    }
                )

            else:
                messages, params = self._get_chat_messages([prompt], stop)
                params = {**params, **kwargs}
                response = completion_with_retry(
                    self, messages=messages, run_manager=run_manager, **params
                )
                if not isinstance(response, dict):
                    response = response.dict()
                choices.extend(response["choices"])
                update_token_usage(_keys, response, token_usage)
        return create_llm_result(choices, prompts, token_usage, self.model_name)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        choices = []
        token_usage: Dict[str, int] = {}
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for prompt in prompts:
            messages = self.prefix_messages + [{"role": "user", "content": prompt}]
            if self.streaming:
                generation: Optional[GenerationChunk] = None
                async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    {
                        "message": {"content": generation.text},
                        "finish_reason": generation.generation_info.get("finish_reason")
                        if generation.generation_info
                        else None,
                        "logprobs": generation.generation_info.get("logprobs")
                        if generation.generation_info
                        else None,
                    }
                )
            else:
                messages, params = self._get_chat_messages([prompt], stop)
                params = {**params, **kwargs}
                response = await acompletion_with_retry(
                    self, messages=messages, run_manager=run_manager, **params
                )
                if not isinstance(response, dict):
                    response = response.dict()
                choices.extend(response["choices"])
                update_token_usage(_keys, response, token_usage)
        return create_llm_result(choices, prompts, token_usage, self.model_name)
