"""Azure OpenAI chat wrapper."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Literal, TypeAlias, TypeVar

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openai.chat_models.base import BaseChatOpenAI

logger = logging.getLogger(__name__)


_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[_BM] | type
_DictOrPydantic: TypeAlias = dict | _BM


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class AzureChatOpenAI(BaseChatOpenAI):
    r"""Azure OpenAI chat model integration.

    Setup:
        Head to the Azure [OpenAI quickstart guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/chatgpt-quickstart?tabs=keyless%2Ctypescript-keyless%2Cpython-new%2Ccommand-line&pivots=programming-language-python)
        to create your Azure OpenAI deployment.

        Then install `langchain-openai` and set environment variables
        `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`:

        ```bash
        pip install -U langchain-openai

        export AZURE_OPENAI_API_KEY="your-api-key"
        export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
        ```

    Key init args — completion params:
        azure_deployment:
            Name of Azure OpenAI deployment to use.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.
        logprobs:
            Whether to return logprobs.

    Key init args — client params:
        api_version:
            Azure OpenAI REST API version to use (distinct from the version of the
            underlying model). [See more on the different versions.](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning)
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        organization:
            OpenAI organization ID. If not passed in will be read from env
            var `OPENAI_ORG_ID`.
        model:
            The name of the underlying OpenAI model. Used for tracing and token
            counting. Does not affect completion. E.g. `'gpt-4'`, `'gpt-35-turbo'`, etc.
        model_version:
            The version of the underlying OpenAI model. Used for tracing and token
            counting. Does not affect completion. E.g., `'0125'`, `'0125-preview'`, etc.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_openai import AzureChatOpenAI

        model = AzureChatOpenAI(
            azure_deployment="your-deployment",
            api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # organization="...",
            # model="gpt-35-turbo",
            # model_version="0125",
            # other params...
        )
        ```

    !!! note
        Any param which is not explicitly supported will be passed directly to the
        `openai.AzureOpenAI.chat.completions.create(...)` API every time to the model is
        invoked.

        For example:

        ```python
        from langchain_openai import AzureChatOpenAI
        import openai

        AzureChatOpenAI(..., logprobs=True).invoke(...)

        # results in underlying API call of:

        openai.AzureOpenAI(..).chat.completions.create(..., logprobs=True)

        # which is also equivalent to:

        AzureChatOpenAI(...).invoke(..., logprobs=True)
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

        ```python
        AIMessage(
            content="J'adore programmer.",
            usage_metadata={
                "input_tokens": 28,
                "output_tokens": 6,
                "total_tokens": 34,
            },
            response_metadata={
                "token_usage": {
                    "completion_tokens": 6,
                    "prompt_tokens": 28,
                    "total_tokens": 34,
                },
                "model_name": "gpt-4",
                "system_fingerprint": "fp_7ec89fabc6",
                "prompt_filter_results": [
                    {
                        "prompt_index": 0,
                        "content_filter_results": {
                            "hate": {"filtered": False, "severity": "safe"},
                            "self_harm": {"filtered": False, "severity": "safe"},
                            "sexual": {"filtered": False, "severity": "safe"},
                            "violence": {"filtered": False, "severity": "safe"},
                        },
                    }
                ],
                "finish_reason": "stop",
                "logprobs": None,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            },
            id="run-6d7a5282-0de0-4f27-9cc0-82a9db9a3ce9-0",
        )
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

        ```python
        AIMessageChunk(content="", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(content="J", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(content="'", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(content="ad", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(content="ore", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(content=" la", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(
            content=" programm", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f"
        )
        AIMessageChunk(content="ation", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(content=".", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
        AIMessageChunk(
            content="",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "gpt-4",
                "system_fingerprint": "fp_811936bd4f",
            },
            id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f",
        )
        ```

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

        ```python
        AIMessageChunk(
            content="J'adore la programmation.",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "gpt-4",
                "system_fingerprint": "fp_811936bd4f",
            },
            id="run-ba60e41c-9258-44b8-8f3a-2f10599643b3",
        )
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetWeather",
                "args": {"location": "Los Angeles, CA"},
                "id": "call_6XswGD5Pqk8Tt5atYr7tfenU",
            },
            {
                "name": "GetWeather",
                "args": {"location": "New York, NY"},
                "id": "call_ZVL15vA8Y7kXqOy3dtmQgeCi",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "call_49CFW8zqC9W7mh7hbMLSIrXw",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "New York, NY"},
                "id": "call_6ghfKxV264jEfe1mRIkS3PE7",
            },
        ]
        ```

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(
                description="How funny the joke is, from 1 to 10"
            )


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup="Why was the cat sitting on the computer?",
            punchline="To keep an eye on the mouse!",
            rating=None,
        )
        ```

        See `AzureChatOpenAI.with_structured_output()` for more.

    JSON mode:
        ```python
        json_model = model.bind(response_format={"type": "json_object"})
        ai_msg = json_model.invoke(
            "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
        )
        ai_msg.content
        ```

        ```python
        '\\n{\\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\\n}'
        ```

    Image input:
        ```python
        import base64
        import httpx
        from langchain_core.messages import HumanMessage

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
        ai_msg = model.invoke([message])
        ai_msg.content
        ```

        ```python
        "The weather in the image appears to be quite pleasant. The sky is mostly clear"
        ```

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```python
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```
    Logprobs:
        ```python
        logprobs_model = model.bind(logprobs=True)
        ai_msg = logprobs_model.invoke(messages)
        ai_msg.response_metadata["logprobs"]
        ```

        ```python
        {
            "content": [
                {
                    "token": "J",
                    "bytes": [74],
                    "logprob": -4.9617593e-06,
                    "top_logprobs": [],
                },
                {
                    "token": "'adore",
                    "bytes": [39, 97, 100, 111, 114, 101],
                    "logprob": -0.25202933,
                    "top_logprobs": [],
                },
                {
                    "token": " la",
                    "bytes": [32, 108, 97],
                    "logprob": -0.20141791,
                    "top_logprobs": [],
                },
                {
                    "token": " programmation",
                    "bytes": [
                        32,
                        112,
                        114,
                        111,
                        103,
                        114,
                        97,
                        109,
                        109,
                        97,
                        116,
                        105,
                        111,
                        110,
                    ],
                    "logprob": -1.9361265e-07,
                    "top_logprobs": [],
                },
                {
                    "token": ".",
                    "bytes": [46],
                    "logprob": -1.2233183e-05,
                    "top_logprobs": [],
                },
            ]
        }
        ```

    Response metadata
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```

        ```python
        {
            "token_usage": {
                "completion_tokens": 6,
                "prompt_tokens": 28,
                "total_tokens": 34,
            },
            "model_name": "gpt-35-turbo",
            "system_fingerprint": None,
            "prompt_filter_results": [
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                }
            ],
            "finish_reason": "stop",
            "logprobs": None,
            "content_filter_results": {
                "hate": {"filtered": False, "severity": "safe"},
                "self_harm": {"filtered": False, "severity": "safe"},
                "sexual": {"filtered": False, "severity": "safe"},
                "violence": {"filtered": False, "severity": "safe"},
            },
        }
        ```
    """  # noqa: E501

    azure_endpoint: str | None = Field(
        default_factory=from_env("AZURE_OPENAI_ENDPOINT", default=None)
    )
    """Your Azure endpoint, including the resource.

        Automatically inferred from env var `AZURE_OPENAI_ENDPOINT` if not provided.

        Example: `https://example-resource.azure.openai.com/`
    """
    deployment_name: str | None = Field(default=None, alias="azure_deployment")
    """A model deployment.

        If given sets the base client URL to include `/deployments/{azure_deployment}`

        !!! note
            This means you won't be able to use non-deployment endpoints.
    """
    openai_api_version: str | None = Field(
        alias="api_version",
        default_factory=from_env("OPENAI_API_VERSION", default=None),
    )
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    # Check OPENAI_API_KEY for backwards compatibility.
    # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
    # other forms of azure credentials.
    openai_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"], default=None
        ),
    )
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    azure_ad_token: SecretStr | None = Field(
        default_factory=secret_from_env("AZURE_OPENAI_AD_TOKEN", default=None)
    )
    """Your Azure Active Directory token.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.

        For more, see [this page](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id).
    """
    azure_ad_token_provider: Callable[[], str] | None = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every sync request. For async requests,
        will be invoked if `azure_ad_async_token_provider` is not provided.
    """

    azure_ad_async_token_provider: Callable[[], Awaitable[str]] | None = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every async request.
    """

    model_version: str = ""
    """The version of the model (e.g. `'0125'` for `'gpt-3.5-0125'`).

    Azure OpenAI doesn't return model version with the response by default so it must
    be manually specified if you want to use this information downstream, e.g. when
    calculating costs.

    When you specify the version, it will be appended to the model name in the
    response. Setting correct version will help you to calculate the cost properly.
    Model version is not validated, so make sure you set it correctly to get the
    correct cost.
    """

    openai_api_type: str | None = Field(
        default_factory=from_env("OPENAI_API_TYPE", default="azure")
    )
    """Legacy, for `openai<1.0.0` support."""

    validate_base_url: bool = True
    """If legacy arg `openai_api_base` is passed in, try to infer if it is a
        `base_url` or `azure_endpoint` and update client params accordingly.
    """

    model_name: str | None = Field(default=None, alias="model")  # type: ignore[assignment]
    """Name of the deployed OpenAI model, e.g. `'gpt-4o'`, `'gpt-35-turbo'`, etc.

    Distinct from the Azure deployment name, which is set by the Azure user.
    Used for tracing and token counting.

    !!! warning
        Does NOT affect completion.
    """

    disabled_params: dict[str, Any] | None = Field(default=None)
    """Parameters of the OpenAI client or chat.completions endpoint that should be
    disabled for the given model.

    Should be specified as `{"param": None | ['val1', 'val2']}` where the key is the
    parameter and the value is either None, meaning that parameter should never be
    used, or it's a list of disabled values for the parameter.

    For example, older models may not support the `'parallel_tool_calls'` parameter at
    all, in which case `disabled_params={"parallel_tool_calls: None}` can ben passed
    in.

    If a parameter is disabled then it will not be used by default in any methods, e.g.
    in
    `langchain_openai.chat_models.azure.AzureChatOpenAI.with_structured_output`.
    However this does not prevent a user from directly passed in the parameter during
    invocation.

    By default, unless `model_name="gpt-4o"` is specified, then
    `'parallel_tools_calls'` will be disabled.
    """

    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")  # type: ignore[assignment]
    """Maximum number of tokens to generate."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "chat_models", "azure_openai"]`
        """
        return ["langchain", "chat_models", "azure_openai"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Get the mapping of secret environment variables."""
        return {
            "openai_api_key": "AZURE_OPENAI_API_KEY",
            "azure_ad_token": "AZURE_OPENAI_AD_TOKEN",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Check if the class is serializable in langchain."""
        return True

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        if self.disabled_params is None:
            # As of 09-17-2024 'parallel_tool_calls' param is only supported for gpt-4o.
            if self.model_name and self.model_name == "gpt-4o":
                pass
            else:
                self.disabled_params = {"parallel_tool_calls": None}

        # Check OPENAI_ORGANIZATION for backwards compatibility.
        self.openai_organization = (
            self.openai_organization
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )

        # Enable stream_usage by default if using default base URL and client
        if all(
            getattr(self, key, None) is None
            for key in (
                "stream_usage",
                "openai_proxy",
                "openai_api_base",
                "base_url",
                "client",
                "root_client",
                "async_client",
                "root_async_client",
                "http_client",
                "http_async_client",
            )
        ):
            self.stream_usage = True

        # For backwards compatibility. Before openai v1, no distinction was made
        # between azure_endpoint and base_url (openai_api_base).
        openai_api_base = self.openai_api_base
        if openai_api_base and self.validate_base_url:
            if "/openai" not in openai_api_base:
                msg = (
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`)."
                )
                raise ValueError(msg)
            if self.deployment_name:
                msg = (
                    "As of openai>=1.0.0, if `azure_deployment` (or alias "
                    "`deployment_name`) is specified then "
                    "`base_url` (or alias `openai_api_base`) should not be. "
                    "If specifying `azure_deployment`/`deployment_name` then use "
                    "`azure_endpoint` instead of `base_url`.\n\n"
                    "For example, you could specify:\n\n"
                    'azure_endpoint="https://xxx.openai.azure.com/", '
                    'azure_deployment="my-deployment"\n\n'
                    "Or you can equivalently specify:\n\n"
                    'base_url="https://xxx.openai.azure.com/openai/deployments/my-deployment"'
                )
                raise ValueError(msg)
        client_params: dict = {
            "api_version": self.openai_api_version,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.deployment_name,
            "api_key": (
                self.openai_api_key.get_secret_value() if self.openai_api_key else None
            ),
            "azure_ad_token": (
                self.azure_ad_token.get_secret_value() if self.azure_ad_token else None
            ),
            "azure_ad_token_provider": self.azure_ad_token_provider,
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": {
                **(self.default_headers or {}),
                "User-Agent": "langchain-partner-python-azure-openai",
            },
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if not self.client:
            sync_specific = {"http_client": self.http_client}
            self.root_client = openai.AzureOpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
            self.client = self.root_client.chat.completions
        if not self.async_client:
            async_specific = {"http_client": self.http_async_client}

            if self.azure_ad_async_token_provider:
                client_params["azure_ad_token_provider"] = (
                    self.azure_ad_async_token_provider
                )

            self.root_async_client = openai.AsyncAzureOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "azure_deployment": self.deployment_name,
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """Get the attributes relevant to tracing."""
        return {
            "openai_api_type": self.openai_api_type,
            "openai_api_version": self.openai_api_version,
        }

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling Azure OpenAI API."""
        params = super()._default_params
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        return params

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "azure"
        if self.model_name:
            if self.model_version and self.model_version not in self.model_name:
                params["ls_model_name"] = (
                    self.model_name + "-" + self.model_version.lstrip("-")
                )
            else:
                params["ls_model_name"] = self.model_name
        elif self.deployment_name:
            params["ls_model_name"] = self.deployment_name
        return params

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        chat_result = super()._create_chat_result(response, generation_info)

        if not isinstance(response, dict):
            response = response.model_dump()
        for res in response["choices"]:
            if res.get("finish_reason", None) == "content_filter":
                msg = (
                    "Azure has not provided the response due to a content filter "
                    "being triggered"
                )
                raise ValueError(msg)

        if "model" in response:
            model = response["model"]
            if self.model_version:
                model = f"{model}-{self.model_version}"

            chat_result.llm_output = chat_result.llm_output or {}
            chat_result.llm_output["model_name"] = model
        if "prompt_filter_results" in response:
            chat_result.llm_output = chat_result.llm_output or {}
            chat_result.llm_output["prompt_filter_results"] = response[
                "prompt_filter_results"
            ]
        for chat_gen, response_choice in zip(
            chat_result.generations, response["choices"], strict=False
        ):
            chat_gen.generation_info = chat_gen.generation_info or {}
            chat_gen.generation_info["content_filter_results"] = response_choice.get(
                "content_filter_results", {}
            )

        return chat_result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get the request payload, using deployment name for Azure Responses API."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # For Azure Responses API, use deployment name instead of model name
        if (
            self._use_responses_api(payload)
            and not payload.get("model")
            and self.deployment_name
        ):
            payload["model"] = self.deployment_name

        return payload

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return super()._stream_responses(*args, **kwargs)
        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - a JSON Schema,
                - a `TypedDict` class,
                - a Pydantic class,
                - or an OpenAI function/tool schema.

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.

            method: The method for steering model generation, one of:

                - `'json_schema'`:
                    Uses OpenAI's [Structured Output API](https://platform.openai.com/docs/guides/structured-outputs).
                    Supported for `'gpt-4o-mini'`, `'gpt-4o-2024-08-06'`, `'o1'`, and later
                    models.
                - `'function_calling'`:
                    Uses OpenAI's tool-calling (formerly called function calling)
                    [API](https://platform.openai.com/docs/guides/function-calling)
                - `'json_mode'`:
                    Uses OpenAI's [JSON mode](https://platform.openai.com/docs/guides/structured-outputs/json-mode).
                    Note that if using JSON mode then you must include instructions for
                    formatting the output into the desired schema into the model call

                Learn more about the differences between the methods and which models
                support which methods [here](https://platform.openai.com/docs/guides/structured-outputs/function-calling-vs-response-format).

            include_raw:
                If `False` then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If `True`
                then both the raw model response (a `BaseMessage`) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:

                - True:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the [supported schemas](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas?api-mode=responses#supported-schemas).
                - False:
                    Input schema will not be validated and model output will not be
                    validated.
                - None:
                    `strict` argument will not be passed to the model.

                If schema is specified via TypedDict or JSON schema, `strict` is not
                enabled by default. Pass `strict=True` to enable it.

                !!! note
                    `strict` can only be non-null if `method` is `'json_schema'`
                    or `'function_calling'`.
            tools:
                A list of tool-like objects to bind to the chat model. Requires that:

                - `method` is `'json_schema'` (default).
                - `strict=True`
                - `include_raw=True`

                If a model elects to call a
                tool, the resulting `AIMessage` in `'raw'` will include tool calls.

                ??? example

                    ```python
                    from langchain.chat_models import init_chat_model
                    from pydantic import BaseModel


                    class ResponseSchema(BaseModel):
                        response: str


                    def get_weather(location: str) -> str:
                        \"\"\"Get weather at a location.\"\"\"
                        pass

                    model = init_chat_model("openai:gpt-4o-mini")

                    structured_model = model.with_structured_output(
                        ResponseSchema,
                        tools=[get_weather],
                        strict=True,
                        include_raw=True,
                    )

                    structured_model.invoke("What's the weather in Boston?")
                    ```

                    ```python
                    {
                        "raw": AIMessage(content="", tool_calls=[...], ...),
                        "parsing_error": None,
                        "parsed": None,
                    }
                    ```

            kwargs: Additional keyword args are passed through to the model.

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        !!! warning "Behavior changed in 0.3.0"
            `method` default changed from "function_calling" to "json_schema".

        !!! warning "Behavior changed in 0.3.12"
            Support for `tools` added.

        !!! warning "Behavior changed in 0.3.21"
            Pass `kwargs` through to the model.

        ??? note "Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=False`, `strict=True`"

            Note, OpenAI has a number of restrictions on what types of schemas can be
            provided if `strict` = True. When using Pydantic, our model cannot
            specify any Field metadata (like min/max constraints) and fields cannot
            have default values.

            See all constraints [here](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas).

            ```python
            from typing import Optional

            from langchain_openai import AzureChatOpenAI
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=..., description="A justification for the answer."
                )


            model = AzureChatOpenAI(
                azure_deployment="...", model="gpt-4o", temperature=0
            )
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )

            # -> AnswerWithJustification(
            #     answer='They weigh the same',
            #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
            # )
            ```

        ??? note "Example: `schema=Pydantic` class, `method='function_calling'`, `include_raw=False`, `strict=False`"

            ```python
            from typing import Optional

            from langchain_openai import AzureChatOpenAI
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=..., description="A justification for the answer."
                )


            model = AzureChatOpenAI(
                azure_deployment="...", model="gpt-4o", temperature=0
            )
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="function_calling"
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )

            # -> AnswerWithJustification(
            #     answer='They weigh the same',
            #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
            # )
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=True`"

            ```python
            from langchain_openai import AzureChatOpenAI
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = AzureChatOpenAI(
                azure_deployment="...", model="gpt-4o", temperature=0
            )
            structured_model = model.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
            #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
            #     'parsing_error': None
            # }
            ```

        ??? note "Example: `schema=TypedDict` class, `method='json_schema'`, `include_raw=False`, `strict=False`"

            ```python
            from typing_extensions import Annotated, TypedDict

            from langchain_openai import AzureChatOpenAI


            class AnswerWithJustification(TypedDict):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: Annotated[
                    str | None, None, "A justification for the answer."
                ]


            model = AzureChatOpenAI(
                azure_deployment="...", model="gpt-4o", temperature=0
            )
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            #     'answer': 'They weigh the same',
            #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
            # }
            ```

        ??? note "Example: `schema=OpenAI` function schema, `method='json_schema'`, `include_raw=False`"

            ```python
            from langchain_openai import AzureChatOpenAI

            oai_schema = {
                'name': 'AnswerWithJustification',
                'description': 'An answer to the user question along with justification for the answer.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'answer': {'type': 'string'},
                        'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                    },
                    'required': ['answer']
                }

                model = AzureChatOpenAI(
                    azure_deployment="...",
                    model="gpt-4o",
                    temperature=0,
                )
                structured_model = model.with_structured_output(oai_schema)

                structured_model.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_mode'`, `include_raw=True`"

            ```python
            from langchain_openai import AzureChatOpenAI
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                answer: str
                justification: str


            model = AzureChatOpenAI(
                azure_deployment="...",
                model="gpt-4o",
                temperature=0,
            )
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            # -> {
            #     'raw': AIMessage(content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'),
            #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
            #     'parsing_error': None
            # }
            ```

        ??? note "Example: `schema=None`, `method='json_mode'`, `include_raw=True`"

            ```python
            structured_model = model.with_structured_output(
                method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            # -> {
            #     'raw': AIMessage(content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'),
            #     'parsed': {
            #         'answer': 'They are both the same weight.',
            #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
            #     },
            #     'parsing_error': None
            # }
            ```

        """  # noqa: E501
        return super().with_structured_output(
            schema, method=method, include_raw=include_raw, strict=strict, **kwargs
        )
