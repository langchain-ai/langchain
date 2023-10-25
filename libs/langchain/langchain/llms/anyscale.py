"""Wrapper around Anyscale Endpoint"""
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

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.openai import (
    BaseOpenAI,
    acompletion_with_retry,
    completion_with_retry,
)
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import Generation, LLMResult
from langchain.schema.output import GenerationChunk
from langchain.utils import get_from_dict_or_env


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

    To use, you should have the environment variable ``ANYSCALE_API_BASE`` and
    ``ANYSCALE_API_KEY``set with your Anyscale Endpoint, or pass it as a named
    parameter to the constructor.

    Example:
        .. code-block:: python
            from langchain.llms import Anyscale
            anyscalellm = Anyscale(anyscale_api_base="ANYSCALE_API_BASE",
                                   anyscale_api_key="ANYSCALE_API_KEY",
                                   model_name="meta-llama/Llama-2-7b-chat-hf")
            # To leverage Ray for parallel processing
            @ray.remote(num_cpus=1)
            def send_query(llm, text):
                resp = llm(text)
                return resp
            futures = [send_query.remote(anyscalellm, text) for text in texts]
            results = ray.get(futures)
    """

    """Key word arguments to pass to the model."""
    anyscale_api_base: Optional[str] = None
    anyscale_api_key: Optional[str] = None

    prefix_messages: List = Field(default_factory=list)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["anyscale_api_base"] = get_from_dict_or_env(
            values, "anyscale_api_base", "ANYSCALE_API_BASE"
        )
        values["anyscale_api_key"] = get_from_dict_or_env(
            values, "anyscale_api_key", "ANYSCALE_API_KEY"
        )
        try:
            import openai

            ## Always create ChatComplete client, replacing the legacy Complete client
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
            "api_key": self.anyscale_api_key,
            "api_base": self.anyscale_api_base,
        }
        return {**openai_creds, **{"model": self.model_name}, **super()._default_params}

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
            token = stream_resp["choices"][0]["delta"].get("content", "")
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
            token = stream_resp["choices"][0]["delta"].get("content", "")
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
                choices.extend(response["choices"])
                update_token_usage(_keys, response, token_usage)
        return create_llm_result(choices, prompts, token_usage, self.model_name)
