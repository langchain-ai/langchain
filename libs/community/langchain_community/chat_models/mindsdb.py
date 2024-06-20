"""MindsDB Endpoint chat wrapper. Relies heavily on ChatOpenAI."""

import requests
from typing import Text, Dict, Set, Optional

from langchain_community.utils.openai import is_openai_v1
from langchain_community.chat_models.openai import ChatOpenAI

from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

DEFAULT_API_BASE = "https://llm.mdb.ai"
DEFAULT_MODEL = "gpt-3.5-turbo"


class ChatAIMind(ChatOpenAI):
    """
    `Minds Endpoint` Chat large language models from MindsDB.

    See https://docs.mdb.ai/ for information about MindsDB and the MindsDB Endpoint.

    To use this chat model, you should have the ``openai`` python package installed, and the
    environment variable ``MINDSDB_API_KEY`` set with your API key.
    Alternatively, you can use the mindsdb_api_key keyword argument.

    The valid parameters that can be passed to this model are:
    - `mindsdb_api_key`: API key for the MindsDB API. As mentioned above, this can be set as an environment variable.
    - `mindsdb_api_base`: Base URL for the MindsDB API. Default is https://llm.mdb.ai. This can be set as an environment variable.
    - `model`: ID of the model to use. Default is s``gpt-3.5-turbo``. See https://docs.mdb.ai/docs/models for a list of supported models.
    - `max_tokens`: This parameter sets the maximum number of tokens that can be generated in the chat completion. It's limited by the model's total allowable context length, which includes both the input and generated tokens.
    - `temperature`: This parameter controls the randomness of the output with values ranging from 0 to 2. A higher value, increases randomness in the output, while a lower value, like 0.1, results in more focused and deterministic output.
    - `stream`: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. 
    - `frequency_penalty`: This setting ranges from -2.0 to 2.0. Positive values make the model less likely to repeat phrases it has already used.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatAIMind
            chat = ChatAIMind(model_name="llama-3-70b")
    """
    @property
    def _llm_type(self) -> Text:
        """Return type of chat model."""
        return "ai-mind-chat"

    @property
    def lc_secrets(self) -> Dict[Text, Text]:
        return {"mindsdb_api_key": "MINDSDB_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    mindsdb_api_key: SecretStr = Field(default=None)
    mindsdb_api_base: str = Field(default=DEFAULT_API_BASE)
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")

    available_models: Optional[Set[str]] = None

    def __init__(
            self, 
            mindsdb_api_key: SecretStr = None, 
            mindsdb_api_base: Text = None, 
            model: Text = None, 
            max_tokens: int = None,
            temperature: float = None,
            stream: bool = None,
            frequency_penalty: float = None,
    ):
        data = {
            "mindsdb_api_key": mindsdb_api_key,
            "mindsdb_api_base": mindsdb_api_base or self.__fields__["mindsdb_api_base"].default,
            "model_name": model or self.__fields__["model_name"].default,
            "max_tokens": max_tokens or self.__fields__["max_tokens"].default,
            "temperature": temperature or self.__fields__["temperature"].default,
            "streaming": stream or self.__fields__["streaming"].default,
            "frequency_penalty": frequency_penalty or self.__fields__["frequency_penalty"].default,
        }
        super().__init__(**data)

    @staticmethod
    def get_available_models(
        mindsdb_api_key: Optional[Text],
        mindsdb_api_base: str = DEFAULT_API_BASE,
    ) -> Set[Text]:
        """
        Get models supported by the MindsDB API.
        """
        models_url = f"{mindsdb_api_base}/models"
        models_response = requests.get(
            models_url,
            headers={
                "Authorization": f"Bearer {mindsdb_api_key}",
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
        """
        Validate that the MindsDB API credentials are provided and create an OpenAI client.
        Further, validate that the chosen model is supported by the MindsDB API.
        """
        # Validate that the API key and base URL are available.
        values["mindsdb_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "mindsdb_api_key",
                "MINDSDB_API_KEY",
            )
        )
        values["mindsdb_api_base"] = get_from_dict_or_env(
            values,
            "mindsdb_api_base",
            "MINDSDB_API_BASE",
            default=DEFAULT_API_BASE,
        )

        # Validate that the `openai` package can be imported.
        try:
            import openai

        except ImportError as e:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`.",
            ) from e

        # Set the client based on the version of the `openai` package that is being used.
        try:
            if is_openai_v1():
                client_params = {
                    "api_key": values["mindsdb_api_key"].get_secret_value(),
                    "base_url": values["mindsdb_api_base"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).chat.completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).chat.completions
            else:
                values["openai_api_base"] = values["mindsdb_api_base"]
                values["openai_api_key"] = values["mindsdb_api_key"].get_secret_value()
                values["client"] = openai.ChatCompletion
        except AttributeError as exc:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`.",
            ) from exc

        # Validate that the chosen model provided is supported.
        if "model_name" not in values.keys():
            values["model_name"] = DEFAULT_MODEL

        model_name = values["model_name"]
        available_models = cls.get_available_models(
            values["mindsdb_api_key"].get_secret_value(),
            values["mindsdb_api_base"],
        )

        if model_name not in available_models:
            raise ValueError(
                f"Model name {model_name} not found in available models: "
                f"{available_models}.",
            )

        values["available_models"] = available_models

        return values

    # TODO: Evaluate need for tiktoken methods
