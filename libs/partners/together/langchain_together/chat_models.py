"""Wrapper around Together AI's Chat Completions API."""

import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from langchain_openai.chat_models.base import BaseChatOpenAI


class ChatTogether(BaseChatOpenAI):
    r"""ChatTogether chat model.

    Setup:
        Install ``langchain-together`` and set environment variable ``TOGETHER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-together
            export TOGETHER_API_KEY="your-api-key"


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
            Together API key. If not passed in will be read from env var OPENAI_API_KEY.

    Instantiate:
        .. code-block:: python

            from langhcain_together import ChatTogether

            llm = ChatTogether(
                model="meta-llama/Llama-3-70b-chat-hf",
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
                    'model_name': 'meta-llama/Llama-3-70b-chat-hf',
                    'system_fingerprint': None,
                    'finish_reason': 'stop',
                    'logprobs': None
                },
                id='run-168dceca-3b8b-4283-94e3-4c739dbc1525-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 9, 'total_tokens': 41})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            content='J' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content="'" id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ad' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ore' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content=' la' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content=' programm' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ation' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='.' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='' response_metadata={'finish_reason': 'stop', 'model_name': 'meta-llama/Llama-3-70b-chat-hf'} id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'


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
                    'model_name': 'meta-llama/Llama-3-70b-chat-hf',
                    'system_fingerprint': None,
                    'finish_reason': 'stop',
                    'logprobs': None
                },
                id='run-09371a11-7f72-4c53-8e7c-9de5c238b34c-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 9, 'total_tokens': 41})

    Tool calling:
        .. code-block:: python

            from langchain_core.pydantic_v1 import BaseModel, Field

            # Only certain models support tool calling, check the together website to confirm compatibility
            llm = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

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

            from langchain_core.pydantic_v1 import BaseModel, Field


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

    JSON mode:
        .. code-block:: python

            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke(
                "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
            )
            ai_msg.content

        .. code-block:: python

            ' {\\n"random_ints": [\\n13,\\n54,\\n78,\\n45,\\n67,\\n90,\\n11,\\n29,\\n84,\\n33\\n]\\n}'

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
                'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'system_fingerprint': None,
                'finish_reason': 'eos',
                'logprobs': None
            }

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"together_api_key": "TOGETHER_API_KEY"}
        """
        return {"together_api_key": "TOGETHER_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "together"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: Dict[str, Any] = {}

        if self.together_api_base:
            attributes["together_api_base"] = self.together_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "together-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "together"
        return params

    model_name: str = Field(default="meta-llama/Llama-3-8b-chat-hf", alias="model")
    """Model name to use."""
    together_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env are `TOGETHER_API_KEY` if not provided."""
    together_api_base: Optional[str] = Field(
        default="https://api.together.ai/v1/", alias="base_url"
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        values["together_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "together_api_key", "TOGETHER_API_KEY")
        )
        values["together_api_base"] = values["together_api_base"] or os.getenv(
            "TOGETHER_API_BASE"
        )

        client_params = {
            "api_key": (
                values["together_api_key"].get_secret_value()
                if values["together_api_key"]
                else None
            ),
            "base_url": values["together_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }

        if not values.get("client"):
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not values.get("async_client"):
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return values
