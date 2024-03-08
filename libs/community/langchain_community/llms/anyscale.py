"""Wrapper around Anyscale Endpoint"""
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.llms.openai import (
    BaseOpenAI,
    acompletion_with_retry,
    completion_with_retry,
)
from langchain_community.utils.openai import is_openai_v1

DEFAULT_BASE_URL = "https://api.endpoints.anyscale.com/v1"
DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


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
    anyscale_api_key: SecretStr = Field(default=None)
    model_name: str = Field(default=DEFAULT_MODEL)

    prefix_messages: List = Field(default_factory=list)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

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
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).completions
            else:
                values["openai_api_base"] = values["anyscale_api_base"]
                values["openai_api_key"] = values["anyscale_api_key"].get_secret_value()
                values["client"] = openai.Completion
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

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: Optional[str] = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")

                generation: Optional[GenerationChunk] = None
                for chunk in self._stream(_prompts[0], stop, run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": generation.generation_info.get("finish_reason")
                        if generation.generation_info
                        else None,
                        "logprobs": generation.generation_info.get("logprobs")
                        if generation.generation_info
                        else None,
                    }
                )
            else:
                response = completion_with_retry(
                    ## THis is the ONLY change from BaseOpenAI()._generate()
                    self,
                    prompt=_prompts[0],
                    run_manager=run_manager,
                    **params,
                )
                if not isinstance(response, dict):
                    # V1 client returns the response in an PyDantic object instead of
                    # dict. For the transition period, we deep convert it to dict.
                    response = response.dict()

                choices.extend(response["choices"])
                update_token_usage(_keys, response, token_usage)
                if not system_fingerprint:
                    system_fingerprint = response.get("system_fingerprint")
        return self.create_llm_result(
            choices,
            prompts,
            params,
            token_usage,
            system_fingerprint=system_fingerprint,
        )

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to OpenAI's endpoint async with k unique prompts."""
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: Optional[str] = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")

                generation: Optional[GenerationChunk] = None
                async for chunk in self._astream(
                    _prompts[0], stop, run_manager, **kwargs
                ):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": generation.generation_info.get("finish_reason")
                        if generation.generation_info
                        else None,
                        "logprobs": generation.generation_info.get("logprobs")
                        if generation.generation_info
                        else None,
                    }
                )
            else:
                response = await acompletion_with_retry(
                    ## THis is the ONLY change from BaseOpenAI()._agenerate()
                    self,
                    prompt=_prompts[0],
                    run_manager=run_manager,
                    **params,
                )
                if not isinstance(response, dict):
                    response = response.dict()
                choices.extend(response["choices"])
                update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(
            choices,
            prompts,
            params,
            token_usage,
            system_fingerprint=system_fingerprint,
        )
