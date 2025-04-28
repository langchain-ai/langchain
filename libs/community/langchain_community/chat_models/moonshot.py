"""Wrapper around Moonshot chat models."""

from typing import Dict

from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    pre_init,
)

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.moonshot import MOONSHOT_SERVICE_URL_BASE, MoonshotCommon


class MoonshotChat(MoonshotCommon, ChatOpenAI):  # type: ignore[misc]
    """Moonshot chat model integration.

    Setup:
        Install ``openai`` and set environment variables ``MOONSHOT_API_KEY``.

        .. code-block:: bash

            pip install openai
            export MOONSHOT_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Moonshot model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        api_key: Optional[str]
            Moonshot API KEY. If not passed in will be read from env var MOONSHOT_API_KEY.
        api_base: Optional[str]
            Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import MoonshotChat

            chat = MoonshotChat(
                temperature=0.5,
                api_key="your-api-key",
                model="moonshot-v1-8k",
                # api_base="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
                ("human", "我喜欢编程。"),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='I like programming.',
                additional_kwargs={},
                response_metadata={
                    'token_usage': {
                        'completion_tokens': 5,
                        'prompt_tokens': 27,
                        'total_tokens': 32
                    },
                    'model_name': 'moonshot-v1-8k',
                    'system_fingerprint': None,
                    'finish_reason': 'stop',
                    'logprobs': None
                },
                id='run-71c03f4e-6628-41d5-beb6-d2559ae68266-0'
            )

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='' additional_kwargs={} response_metadata={} id='run-80d77096-8b83-4c39-a84d-71d9c746da92'
            content='I' additional_kwargs={} response_metadata={} id='run-80d77096-8b83-4c39-a84d-71d9c746da92'
            content=' like' additional_kwargs={} response_metadata={} id='run-80d77096-8b83-4c39-a84d-71d9c746da92'
            content=' programming' additional_kwargs={} response_metadata={} id='run-80d77096-8b83-4c39-a84d-71d9c746da92'
            content='.' additional_kwargs={} response_metadata={} id='run-80d77096-8b83-4c39-a84d-71d9c746da92'
            content='' additional_kwargs={} response_metadata={'finish_reason': 'stop'} id='run-80d77096-8b83-4c39-a84d-71d9c746da92'

        .. code-block:: python

            stream = chat.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content='I like programming.',
                additional_kwargs={},
                response_metadata={'finish_reason': 'stop'},
                id='run-10c80976-7aa5-4ff7-ba3e-1251665557ef'
            )

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

            # stream:
            # async for chunk in chat.astream(messages):
            #    print(chunk)

            # batch:
            # await chat.abatch([messages])

        .. code-block:: python

            [AIMessage(content='I like programming.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 27, 'total_tokens': 32}, 'model_name': 'moonshot-v1-8k', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-2938b005-9204-4b9f-b273-1c3272fce9e5-0')]

    Response metadata
        .. code-block:: python

            ai_msg = chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'token_usage': {
                    'completion_tokens': 5,
                    'prompt_tokens': 27,
                    'total_tokens': 32
                },
                'model_name': 'moonshot-v1-8k',
                'system_fingerprint': None,
                'finish_reason': 'stop',
                'logprobs': None
            }

    """  # noqa: E501

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the environment is set up correctly."""
        values["moonshot_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                ["moonshot_api_key", "api_key", "openai_api_key"],
                "MOONSHOT_API_KEY",
            )
        )

        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        client_params = {
            "api_key": values["moonshot_api_key"].get_secret_value(),
            "base_url": values["base_url"]
            if "base_url" in values
            else MOONSHOT_SERVICE_URL_BASE,
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(
                **client_params
            ).chat.completions

        return values
