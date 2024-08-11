"""OctoAI Endpoints chat wrapper. Relies heavily on ChatOpenAI."""

from typing import Dict

from langchain_core.pydantic_v1 import Field, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://text.octoai.run/v1/"
DEFAULT_MODEL = "llama-2-13b-chat"


class ChatOctoAI(ChatOpenAI):
    """OctoAI Chat large language models.

    See https://octo.ai/ for information about OctoAI.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``OCTOAI_API_TOKEN`` set with your API token.
    Alternatively, you can use the octoai_api_token keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatOctoAI
            chat = ChatOctoAI(model_name="mixtral-8x7b-instruct")
    """

    octoai_api_base: str = Field(default=DEFAULT_API_BASE)
    octoai_api_token: SecretStr = Field(default=None)
    model_name: str = Field(default=DEFAULT_MODEL)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "octoai-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"octoai_api_token": "OCTOAI_API_TOKEN"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["octoai_api_base"] = get_from_dict_or_env(
            values,
            "octoai_api_base",
            "OCTOAI_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["octoai_api_token"] = convert_to_secret_str(
            get_from_dict_or_env(values, "octoai_api_token", "OCTOAI_API_TOKEN")
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
                    "api_key": values["octoai_api_token"].get_secret_value(),
                    "base_url": values["octoai_api_base"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).chat.completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).chat.completions
            else:
                values["openai_api_base"] = values["octoai_api_base"]
                values["openai_api_key"] = values["octoai_api_token"].get_secret_value()
                values["client"] = openai.ChatCompletion
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        return values
