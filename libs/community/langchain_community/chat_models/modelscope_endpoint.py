"""Wrapper around modelscope chat endpoint models."""

from typing import Dict

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import model_validator

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.modelscope_endpoint import (
    MODELSCOPE_SERVICE_URL_BASE,
    ModelScopeCommon,
)


class ModelScopeChatEndpoint(ModelScopeCommon, ChatOpenAI):  # type: ignore[misc, override, override]
    """Modelscope chat model inference api integration. To use, must have a modelscope account and a modelscope sdk token.
    Refer to https://modelscope.cn/docs/model-service/API-Inference/intro for more details.

    Setup:
        Install ``openai`` and set environment variables ``MODELSCOPE_SDK_TOKEN``.

        .. code-block:: bash

            pip install openai
            export MODELSCOPE_SDK_TOKEN="your-modelscope-sdk-token"

    Key init args — completion params:
        model: str
            Name of Modelscope model to use. Refer to https://modelscope.cn/docs/model-service/API-Inference/intro for available models.
        temperature: Optional[float]
            Sampling temperature, defaults to 0.3.
        max_tokens: Optional[int]
            Max number of tokens to generate, defaults to 1024.

    Key init args — client params:
        modelscope_sdk_token: Optional[str]
            Modelscope SDK Token. If not passed in will be read from env var MODELSCOPE_SDK_TOKEN.
        api_base: Optional[str]
            Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ModelscopeChatEndpoint

            chat = ModelscopeChatEndpoint(
                modelscope_sdk_token="your-modelscope-sdk-token",
                model="Qwen/Qwen2.5-Coder-32B-Instruct",
                temperature=0.5,
                # api_base="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "Write a quick sort code."),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='def quick_sort(arr): ...',
                additional_kwargs={},
                response_metadata={
                    'token_usage': {
                        'completion_tokens': 312,
                        'prompt_tokens': 27,
                        'total_tokens': 339
                    },
                    'model_name': 'Qwen/Qwen2.5-Coder-32B-Instruct',
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

    """  # noqa: E501

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the environment is set up correctly."""
        values["modelscope_sdk_token"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                ["modelscope_sdk_token", "api_key"],
                "MODELSCOPE_SDK_TOKEN",
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
            "api_key": values["modelscope_sdk_token"].get_secret_value(),
            "base_url": values["base_url"]
            if "base_url" in values
            else MODELSCOPE_SERVICE_URL_BASE,
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(
                **client_params
            ).chat.completions

        return values
