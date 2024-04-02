from typing import Any, Dict

from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.llms.openai import BaseOpenAI
from langchain_community.utils.openai import is_openai_v1

DEFAULT_BASE_URL = "https://text.octoai.run/v1/"
DEFAULT_MODEL = "codellama-7b-instruct"


class OctoAIEndpoint(BaseOpenAI):
    """OctoAI LLM Endpoints - OpenAI compatible.

    OctoAIEndpoint is a class to interact with OctoAI Compute Service large
    language model endpoints.

    To use, you should have the environment variable ``OCTOAI_API_TOKEN`` set
    with your API token, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms.octoai_endpoint  import OctoAIEndpoint

            llm = OctoAIEndpoint(
                model="llama-2-13b-chat-fp16",
                max_tokens=200,
                presence_penalty=0,
                temperature=0.1,
                top_p=0.9,
            )

    """

    """Key word arguments to pass to the model."""
    octoai_api_base: str = Field(default=DEFAULT_BASE_URL)
    octoai_api_token: SecretStr = Field(default=None)
    model_name: str = Field(default=DEFAULT_MODEL)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""

        params: Dict[str, Any] = {
            "model": self.model_name,
            **self._default_params,
        }
        if not is_openai_v1():
            params.update(
                {
                    "api_key": self.octoai_api_token.get_secret_value(),
                    "api_base": self.octoai_api_base,
                }
            )

        return {**params, **super()._invocation_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "octoai_endpoint"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["octoai_api_base"] = get_from_dict_or_env(
            values,
            "octoai_api_base",
            "OCTOAI_API_BASE",
            default=DEFAULT_BASE_URL,
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
                    values["client"] = openai.OpenAI(**client_params).completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).completions
            else:
                values["openai_api_base"] = values["octoai_api_base"]
                values["openai_api_key"] = values["octoai_api_token"].get_secret_value()
                values["client"] = openai.Completion
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        if "endpoint_url" in values["model_kwargs"]:
            raise ValueError(
                "`endpoint_url` was deprecated, please use `octoai_api_base`."
            )

        return values
