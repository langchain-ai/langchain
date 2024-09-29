import secrets
from typing import Any, Text

from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://mdb.ai"
DEFAULT_MODEL = "gpt-3.5-turbo"


class BaseMindWrapper(BaseModel):
    minds_api_key: SecretStr = Field(default=None)
    minds_api_base: Text = Field(default=DEFAULT_API_BASE)
    model: Text = Field(default=DEFAULT_MODEL)
    name: Text = Field(default=None)
    client: Any = Field(default=None, exclude=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Validate that the API key and base URL are available.
        self.minds_api_key = convert_to_secret_str(
            get_from_dict_or_env(
                data,
                "minds_api_key",
                "MINDS_API_KEY",
            )
        )
        self.minds_api_base = get_from_dict_or_env(
            data,
            "minds_api_base",
            "MINDS_API_BASE",
            default=DEFAULT_API_BASE,
        )

        # If a name is not provided, generate a random one.
        if not self.name:
            self.name = f"lc_mind_{secrets.token_hex(5)}"

        # Validate that the `openai` package can be imported.
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`.",
            ) from e

        # Set the client based on the version of the `openai` package.
        try:
            if is_openai_v1():
                client_params = {
                    "api_key": self.minds_api_key.get_secret_value(),
                    "base_url": self.minds_api_base,
                }
                if not self.client:
                    self.client = openai.OpenAI(**client_params).chat.completions
            else:
                self.openai_api_base = self.minds_api_base
                self.openai_api_key = self.minds_api_key.get_secret_value()
                self.client = openai.ChatCompletion
        except AttributeError as exc:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`.",
            ) from exc

    def run(self, query: Text) -> Text:
        raise NotImplementedError("Method `run` must be implemented in a subclass.")
