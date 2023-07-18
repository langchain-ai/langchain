"""Wrapper around OpenAI APIs."""
from __future__ import annotations

import logging
import sys
import warnings
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from pydantic import Field, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import BaseLLM, create_base_retry_decorator
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


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


def _update_response(response: Dict[str, Any], stream_response: Dict[str, Any]) -> None:
    """Update response from the stream response."""
    response["choices"][0]["text"] += stream_response["choices"][0]["text"]
    response["choices"][0]["finish_reason"] = stream_response["choices"][0].get(
        "finish_reason", None
    )
    response["choices"][0]["logprobs"] = stream_response["choices"][0]["logprobs"]


def _streaming_response_template() -> Dict[str, Any]:
    return {
        "choices": [
            {
                "text": "",
                "finish_reason": None,
                "logprobs": None,
            }
        ]
    }


def _create_retry_decorator(llm: Union[BaseOpenAI, OpenAIChat]) -> Callable[[Any], Any]:
    import openai

    errors = [
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)


def completion_with_retry(llm: Union[BaseOpenAI, OpenAIChat], **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return llm.client.create(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    llm: Union[BaseOpenAI, OpenAIChat], **kwargs: Any
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


class BaseOpenAI(BaseLLM):
    """Wrapper around OpenAI large language models."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = Field("text-davinci-003", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    n: int = 1
    """How many completions to generate for each prompt."""
    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_organization: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """Adjust the probability of specific tokens being generated."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """Set of special tokens that are allowed。"""
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """Set of special tokens that are not allowed。"""
    tiktoken_model_name: Optional[str] = None
    """The model name to pass to tiktoken when using this class. 
    Tiktoken is used to count the number of tokens in documents to constrain 
    them to be under a certain limit. By default, when set to None, this will 
    be the same as the embedding model name. However, there are some cases 
    where you may want to use this Embedding class with a model name not 
    supported by tiktoken. This can include when using Azure embeddings or 
    when using one of the many model providers that expose an OpenAI-like 
    API but with different models. In those cases, in order to avoid erroring 
    when tiktoken is called, you can specify a model name to use here."""

    def __new__(cls, **data: Any) -> Union[OpenAIChat, BaseOpenAI]:  # type: ignore
        """Initialize the OpenAI object."""
        model_name = data.get("model_name", "")
        if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
            warnings.warn(
                "You are trying to use a chat model. This way of initializing it is "
                "no longer supported. Instead, please use: "
                "`from langchain.chat_models import ChatOpenAI`"
            )
            return OpenAIChat(**data)
        return super().__new__(cls)

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = cls._all_required_field_names()
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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        values["openai_organization"] = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

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
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "request_timeout": self.request_timeout,
            "logit_bias": self.logit_bias,
        }

        # Azure gpt-35-turbo doesn't support best_of
        # don't specify best_of if it is 1
        if self.best_of > 1:
            normal_params["best_of"] = self.best_of

        return {**normal_params, **self.model_kwargs}

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
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")
                params["stream"] = True
                response = _streaming_response_template()
                for stream_resp in completion_with_retry(
                    self, prompt=_prompts, **params
                ):
                    if run_manager:
                        run_manager.on_llm_new_token(
                            stream_resp["choices"][0]["text"],
                            verbose=self.verbose,
                            logprobs=stream_resp["choices"][0]["logprobs"],
                        )
                    _update_response(response, stream_resp)
                choices.extend(response["choices"])
            else:
                response = completion_with_retry(self, prompt=_prompts, **params)
                choices.extend(response["choices"])
            if not self.streaming:
                # Can't update token usage if streaming
                update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(choices, prompts, token_usage)

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
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")
                params["stream"] = True
                response = _streaming_response_template()
                async for stream_resp in await acompletion_with_retry(
                    self, prompt=_prompts, **params
                ):
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            stream_resp["choices"][0]["text"],
                            verbose=self.verbose,
                            logprobs=stream_resp["choices"][0]["logprobs"],
                        )
                    _update_response(response, stream_resp)
                choices.extend(response["choices"])
            else:
                response = await acompletion_with_retry(self, prompt=_prompts, **params)
                choices.extend(response["choices"])
            if not self.streaming:
                # Can't update token usage if streaming
                update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(choices, prompts, token_usage)

    def get_sub_prompts(
        self,
        params: Dict[str, Any],
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Get the sub prompts for llm call."""
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
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
            sub_choices = choices[i * self.n : (i + 1) * self.n]
            generations.append(
                [
                    Generation(
                        text=choice["text"],
                        generation_info=dict(
                            finish_reason=choice.get("finish_reason"),
                            logprobs=choice.get("logprobs"),
                        ),
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return LLMResult(generations=generations, llm_output=llm_output)

    def stream(self, prompt: str, stop: Optional[List[str]] = None) -> Generator:
        """Call OpenAI with streaming flag and return the resulting generator.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens from OpenAI.

        Example:
            .. code-block:: python

                generator = openai.stream("Tell me a joke.")
                for token in generator:
                    yield token
        """
        params = self.prep_streaming_params(stop)
        generator = self.client.create(prompt=prompt, **params)

        return generator

    def prep_streaming_params(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Prepare the params for streaming."""
        params = self._invocation_params
        if "best_of" in params and params["best_of"] != 1:
            raise ValueError("OpenAI only supports best_of == 1 for streaming")
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        params["stream"] = True
        return params

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        openai_creds: Dict[str, Any] = {
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
            "organization": self.openai_organization,
        }
        if self.openai_proxy:
            import openai

            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}  # type: ignore[assignment]  # noqa: E501
        return {**openai_creds, **self._default_params}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai"

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token IDs using the tiktoken package."""
        # tiktoken NOT supported for Python < 3.8
        if sys.version_info[1] < 8:
            return super().get_num_tokens(text)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please install it with `pip install tiktoken`."
            )

        model_name = self.tiktoken_model_name or self.model_name
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            enc = tiktoken.get_encoding(model)

        return enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    @staticmethod
    def modelname_to_contextsize(modelname: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a model.

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size

        Example:
            .. code-block:: python

                max_tokens = openai.modelname_to_contextsize("text-davinci-003")
        """
        model_token_mapping = {
            "gpt-4": 8192,
            "gpt-4-0314": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0314": 32768,
            "gpt-4-32k-0613": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0301": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-16k-0613": 16385,
            "text-ada-001": 2049,
            "ada": 2049,
            "text-babbage-001": 2040,
            "babbage": 2049,
            "text-curie-001": 2049,
            "curie": 2049,
            "davinci": 2049,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "code-davinci-002": 8001,
            "code-davinci-001": 8001,
            "code-cushman-002": 2048,
            "code-cushman-001": 2048,
        }

        # handling finetuned models
        if "ft-" in modelname:
            modelname = modelname.split(":")[0]

        context_size = model_token_mapping.get(modelname, None)

        if context_size is None:
            raise ValueError(
                f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
                "Known models are: " + ", ".join(model_token_mapping.keys())
            )

        return context_size

    @property
    def max_context_size(self) -> int:
        """Get max context size for this model."""
        return self.modelname_to_contextsize(self.model_name)

    def max_tokens_for_prompt(self, prompt: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a prompt.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The maximum number of tokens to generate for a prompt.

        Example:
            .. code-block:: python

                max_tokens = openai.max_token_for_prompt("Tell me a joke.")
        """
        num_tokens = self.get_num_tokens(prompt)
        return self.max_context_size - num_tokens


class OpenAI(BaseOpenAI):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            openai = OpenAI(model_name="text-davinci-003")
    """

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}


class AzureOpenAI(BaseOpenAI):
    """Wrapper around Azure-specific OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import AzureOpenAI
            openai = AzureOpenAI(model_name="text-davinci-003")
    """

    deployment_name: str = ""
    """Deployment name to use."""
    openai_api_type: str = ""
    openai_api_version: str = ""

    @root_validator()
    def validate_azure_settings(cls, values: Dict) -> Dict:
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", "azure"
        )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            **{"deployment_name": self.deployment_name},
            **super()._identifying_params,
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        openai_params = {
            "engine": self.deployment_name,
            "api_type": self.openai_api_type,
            "api_version": self.openai_api_version,
        }
        return {**openai_params, **super()._invocation_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azure"


class OpenAIChat(BaseLLM):
    """Wrapper around OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAIChat
            openaichat = OpenAIChat(model_name="gpt-3.5-turbo")
    """

    client: Any  #: :meta private:
    model_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""
    streaming: bool = False
    """Whether to stream the results or not."""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """Set of special tokens that are allowed。"""
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """Set of special tokens that are not allowed。"""

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        openai_proxy = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        openai_organization = get_from_dict_or_env(
            values, "openai_organization", "OPENAI_ORGANIZATION", default=""
        )
        try:
            import openai

            openai.api_key = openai_api_key
            if openai_api_base:
                openai.api_base = openai_api_base
            if openai_organization:
                openai.organization = openai_organization
            if openai_proxy:
                openai.proxy = {"http": openai_proxy, "https": openai_proxy}  # type: ignore[assignment]  # noqa: E501
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        warnings.warn(
            "You are trying to use a chat model. This way of initializing it is "
            "no longer supported. Instead, please use: "
            "`from langchain.chat_models import ChatOpenAI`"
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return self.model_kwargs

    def _get_chat_params(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> Tuple:
        if len(prompts) > 1:
            raise ValueError(
                f"OpenAIChat currently only supports single prompt, got {prompts}"
            )
        messages = self.prefix_messages + [{"role": "user", "content": prompts[0]}]
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params.get("max_tokens") == -1:
            # for ChatGPT api, omitting max_tokens is equivalent to having no limit
            del params["max_tokens"]
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
        if self.streaming:
            response = ""
            params["stream"] = True
            for stream_resp in completion_with_retry(self, messages=messages, **params):
                token = stream_resp["choices"][0]["delta"].get("content", "")
                response += token
                if run_manager:
                    run_manager.on_llm_new_token(
                        token,
                    )
            return LLMResult(
                generations=[[Generation(text=response)]],
            )
        else:
            full_response = completion_with_retry(self, messages=messages, **params)
            llm_output = {
                "token_usage": full_response["usage"],
                "model_name": self.model_name,
            }
            return LLMResult(
                generations=[
                    [Generation(text=full_response["choices"][0]["message"]["content"])]
                ],
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
        if self.streaming:
            response = ""
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(
                self, messages=messages, **params
            ):
                token = stream_resp["choices"][0]["delta"].get("content", "")
                response += token
                if run_manager:
                    await run_manager.on_llm_new_token(
                        token,
                    )
            return LLMResult(
                generations=[[Generation(text=response)]],
            )
        else:
            full_response = await acompletion_with_retry(
                self, messages=messages, **params
            )
            llm_output = {
                "token_usage": full_response["usage"],
                "model_name": self.model_name,
            }
            return LLMResult(
                generations=[
                    [Generation(text=full_response["choices"][0]["message"]["content"])]
                ],
                llm_output=llm_output,
            )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai-chat"

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token IDs using the tiktoken package."""
        # tiktoken NOT supported for Python < 3.8
        if sys.version_info[1] < 8:
            return super().get_token_ids(text)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please install it with `pip install tiktoken`."
            )

        enc = tiktoken.encoding_for_model(self.model_name)
        return enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
