"""Azure OpenAI chat wrapper."""

from __future__ import annotations

import logging
import os
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openai.chat_models.base import BaseChatOpenAI

logger = logging.getLogger(__name__)


_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]
_DictOrPydantic = Union[Dict, _BM]


class _AllReturnType(TypedDict):
    raw: BaseMessage
    parsed: Optional[_DictOrPydantic]
    parsing_error: Optional[BaseException]


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class AzureChatOpenAI(BaseChatOpenAI):
    """Azure OpenAI chat model integration.

    Setup:
        Head to the https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python
        to create your Azure OpenAI deployment.

        Then install ``langchain-openai`` and set environment variables
        ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT``:

        .. code-block:: bash

            pip install -U langchain-openai

            export AZURE_OPENAI_API_KEY="your-api-key"
            export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

    Key init args — completion params:
        azure_deployment: str
            Name of Azure OpenAI deployment to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        logprobs: Optional[bool]
            Whether to return logprobs.

    Key init args — client params:
        api_version: str
            Azure OpenAI API version to use. See more on the different versions here:
            https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        timeout: Union[float, Tuple[float, float], Any, None]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        organization: Optional[str]
            OpenAI organization ID. If not passed in will be read from env
            var OPENAI_ORG_ID.
        model: Optional[str]
            The name of the underlying OpenAI model. Used for tracing and token
            counting. Does not affect completion. E.g. "gpt-4", "gpt-35-turbo", etc.
        model_version: Optional[str]
            The version of the underlying OpenAI model. Used for tracing and token
            counting. Does not affect completion. E.g., "0125", "0125-preview", etc.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_openai import AzureChatOpenAI

            llm = AzureChatOpenAI(
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

    **NOTE**: Any param which is not explicitly supported will be passed directly to the
    ``openai.AzureOpenAI.chat.completions.create(...)`` API every time to the model is
    invoked. For example:
        .. code-block:: python

            from langchain_openai import AzureChatOpenAI
            import openai

            AzureChatOpenAI(..., logprobs=True).invoke(...)

            # results in underlying API call of:

            openai.AzureOpenAI(..).chat.completions.create(..., logprobs=True)

            # which is also equivalent to:

            AzureChatOpenAI(...).invoke(..., logprobs=True)

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful translator. Translate the user sentence to French.",
                ),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(
                content="J'adore programmer.",
                usage_metadata={"input_tokens": 28, "output_tokens": 6, "total_tokens": 34},
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

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content="", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
            AIMessageChunk(content="J", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
            AIMessageChunk(content="'", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
            AIMessageChunk(content="ad", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
            AIMessageChunk(content="ore", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
            AIMessageChunk(content=" la", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
            AIMessageChunk(content=" programm", id="run-a6f294d3-0700-4f6a-abc2-c6ef1178c37f")
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

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content="J'adore la programmation.",
                response_metadata={
                    "finish_reason": "stop",
                    "model_name": "gpt-4",
                    "system_fingerprint": "fp_811936bd4f",
                },
                id="run-ba60e41c-9258-44b8-8f3a-2f10599643b3",
            )

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

    Tool calling:
        .. code-block:: python

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


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python

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

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(
                setup="Why was the cat sitting on the computer?",
                punchline="To keep an eye on the mouse!",
                rating=None,
            )

        See ``AzureChatOpenAI.with_structured_output()`` for more.

    JSON mode:
        .. code-block:: python

            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke(
                "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
            )
            ai_msg.content

        .. code-block:: python

            '\\n{\\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\\n}'

    Image input:
        .. code-block:: python

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
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            "The weather in the image appears to be quite pleasant. The sky is mostly clear"

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}

    Logprobs:
        .. code-block:: python

            logprobs_llm = llm.bind(logprobs=True)
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

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

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

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
    """  # noqa: E501

    azure_endpoint: Optional[str] = Field(
        default_factory=from_env("AZURE_OPENAI_ENDPOINT", default=None)
    )
    """Your Azure endpoint, including the resource.

        Automatically inferred from env var `AZURE_OPENAI_ENDPOINT` if not provided.

        Example: `https://example-resource.azure.openai.com/`
    """
    deployment_name: Union[str, None] = Field(default=None, alias="azure_deployment")
    """A model deployment. 
    
        If given sets the base client URL to include `/deployments/{azure_deployment}`.
        Note: this means you won't be able to use non-deployment endpoints.
    """
    openai_api_version: Optional[str] = Field(
        alias="api_version",
        default_factory=from_env("OPENAI_API_VERSION", default=None),
    )
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    # Check OPENAI_API_KEY for backwards compatibility.
    # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
    # other forms of azure credentials.
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"], default=None
        ),
    )
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    azure_ad_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AZURE_OPENAI_AD_TOKEN", default=None)
    )
    """Your Azure Active Directory token.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.

        For more:
        https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id.
    """
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    """A function that returns an Azure Active Directory token.
        
        Will be invoked on every sync request. For async requests,
        will be invoked if `azure_ad_async_token_provider` is not provided.
    """

    azure_ad_async_token_provider: Union[Callable[[], Awaitable[str]], None] = None
    """A function that returns an Azure Active Directory token.
        
        Will be invoked on every async request.
    """

    model_version: str = ""
    """The version of the model (e.g. "0125" for gpt-3.5-0125).

    Azure OpenAI doesn't return model version with the response by default so it must 
    be manually specified if you want to use this information downstream, e.g. when
    calculating costs.

    When you specify the version, it will be appended to the model name in the 
    response. Setting correct version will help you to calculate the cost properly. 
    Model version is not validated, so make sure you set it correctly to get the 
    correct cost.
    """

    openai_api_type: Optional[str] = Field(
        default_factory=from_env("OPENAI_API_TYPE", default="azure")
    )
    """Legacy, for openai<1.0.0 support."""

    validate_base_url: bool = True
    """If legacy arg openai_api_base is passed in, try to infer if it is a base_url or 
        azure_endpoint and update client params accordingly.
    """

    model_name: Optional[str] = Field(default=None, alias="model")  # type: ignore[assignment]
    """Name of the deployed OpenAI model, e.g. "gpt-4o", "gpt-35-turbo", etc. 
    
    Distinct from the Azure deployment name, which is set by the Azure user.
    Used for tracing and token counting. Does NOT affect completion.
    """

    disabled_params: Optional[Dict[str, Any]] = Field(default=None)
    """Parameters of the OpenAI client or chat.completions endpoint that should be 
    disabled for the given model.

    Should be specified as ``{"param": None | ['val1', 'val2']}`` where the key is the 
    parameter and the value is either None, meaning that parameter should never be
    used, or it's a list of disabled values for the parameter.

    For example, older models may not support the 'parallel_tool_calls' parameter at 
    all, in which case ``disabled_params={"parallel_tool_calls: None}`` can ben passed 
    in.
    
    If a parameter is disabled then it will not be used by default in any methods, e.g.
    in 
    :meth:`~langchain_openai.chat_models.azure.AzureChatOpenAI.with_structured_output`.
    However this does not prevent a user from directly passed in the parameter during
    invocation. 
    
    By default, unless ``model_name="gpt-4o"`` is specified, then 
    'parallel_tools_calls' will be disabled.
    """

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "azure_openai"]

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "openai_api_key": "AZURE_OPENAI_API_KEY",
            "azure_ad_token": "AZURE_OPENAI_AD_TOKEN",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

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
        # For backwards compatibility. Before openai v1, no distinction was made
        # between azure_endpoint and base_url (openai_api_base).
        openai_api_base = self.openai_api_base
        if openai_api_base and self.validate_base_url:
            if "/openai" not in openai_api_base:
                raise ValueError(
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`)."
                )
            if self.deployment_name:
                raise ValueError(
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
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
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
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"azure_deployment": self.deployment_name},
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        return {
            "openai_api_type": self.openai_api_type,
            "openai_api_version": self.openai_api_version,
        }

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
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
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        chat_result = super()._create_chat_result(response, generation_info)

        if not isinstance(response, dict):
            response = response.model_dump()
        for res in response["choices"]:
            if res.get("finish_reason", None) == "content_filter":
                raise ValueError(
                    "Azure has not provided the response due to a content filter "
                    "being triggered"
                )

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
            chat_result.generations, response["choices"]
        ):
            chat_gen.generation_info = chat_gen.generation_info or {}
            chat_gen.generation_info["content_filter_results"] = response_choice.get(
                "content_filter_results", {}
            )

        return chat_result
