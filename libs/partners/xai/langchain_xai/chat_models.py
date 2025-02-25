"""Wrapper around xAI's Chat Completions API."""

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.utils import secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class ChatXAI(BaseChatOpenAI):  # type: ignore[override]
    r"""ChatXAI chat model.

    Setup:
        Install ``langchain-xai`` and set environment variable ``XAI_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-xai
            export XAI_API_KEY="your-api-key"


    Key init args — completion params:
        model: str
            Name of model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        logprobs: Optional[bool]
            Whether to return logprobs.

    Key init args — client params:
        timeout: Union[float, Tuple[float, float], Any, None]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            xAI API key. If not passed in will be read from env var `XAI_API_KEY`.

    Instantiate:
        .. code-block:: python

            from langchain_xai import ChatXAI

            llm = ChatXAI(
                model="grok-beta",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

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
                content="J'adore la programmation.",
                response_metadata={
                    'token_usage': {'completion_tokens': 9, 'prompt_tokens': 32, 'total_tokens': 41},
                    'model_name': 'grok-beta',
                    'system_fingerprint': None,
                    'finish_reason': 'stop',
                    'logprobs': None
                },
                id='run-168dceca-3b8b-4283-94e3-4c739dbc1525-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 9, 'total_tokens': 41})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.text(), end="")

        .. code-block:: python

            content='J' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content="'" id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ad' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ore' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content=' la' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content=' programm' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ation' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='.' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='' response_metadata={'finish_reason': 'stop', 'model_name': 'grok-beta'} id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'


    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={
                    'token_usage': {'completion_tokens': 9, 'prompt_tokens': 32, 'total_tokens': 41},
                    'model_name': 'grok-beta',
                    'system_fingerprint': None,
                    'finish_reason': 'stop',
                    'logprobs': None
                },
                id='run-09371a11-7f72-4c53-8e7c-9de5c238b34c-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 9, 'total_tokens': 41})

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            llm = ChatXAI(model="grok-beta")

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
                "Which city is bigger: LA or NY?"
            )
            ai_msg.tool_calls


        .. code-block:: python

            [
                {
                    'name': 'GetPopulation',
                    'args': {'location': 'NY'},
                    'id': 'call_m5tstyn2004pre9bfuxvom8x',
                    'type': 'tool_call'
                },
                {
                    'name': 'GetPopulation',
                    'args': {'location': 'LA'},
                    'id': 'call_0vjgq455gq1av5sp9eb1pw6a',
                    'type': 'tool_call'
                }
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
                setup='Why was the cat sitting on the computer?',
                punchline='To keep an eye on the mouse!',
                rating=7
            )

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 37, 'output_tokens': 6, 'total_tokens': 43}

    Logprobs:
        .. code-block:: python

            logprobs_llm = llm.bind(logprobs=True)
            messages=[("human","Say Hello World! Do not return anything else.")]
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

            {
                'content': None,
                'token_ids': [22557, 3304, 28808, 2],
                'tokens': [' Hello', ' World', '!', '</s>'],
                'token_logprobs': [-4.7683716e-06, -5.9604645e-07, 0, -0.057373047]
            }


    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'token_usage': {
                    'completion_tokens': 4,
                    'prompt_tokens': 19,
                    'total_tokens': 23
                    },
                'model_name': 'grok-beta',
                'system_fingerprint': None,
                'finish_reason': 'stop',
                'logprobs': None
            }

    """  # noqa: E501

    model_name: str = Field(alias="model")
    """Model name to use."""
    xai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("XAI_API_KEY", default=None),
    )
    """xAI API key.

    Automatically read from env variable `XAI_API_KEY` if not provided.
    """
    xai_api_base: str = Field(default="https://api.x.ai/v1/")
    """Base URL path for API requests."""

    openai_api_key: Optional[SecretStr] = None
    openai_api_base: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"xai_api_key": "XAI_API_KEY"}
        """
        return {"xai_api_key": "XAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_xai", "chat_models"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: Dict[str, Any] = {}

        if self.xai_api_base:
            attributes["xai_api_base"] = self.xai_api_base

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "xai-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "xai"
        return params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params: dict = {
            "api_key": (
                self.xai_api_key.get_secret_value() if self.xai_api_key else None
            ),
            "base_url": self.xai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            raise ValueError(
                "xAI API key is not set. Please set it in the `xai_api_key` field or "
                "in the `XAI_API_KEY` environment variable."
            )

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
        return self
