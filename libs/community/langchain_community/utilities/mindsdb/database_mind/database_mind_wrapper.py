from openai import OpenAI
from typing import Text, Dict, Any, Literal
from mindsdb_sdk.utils.mind import Mind, create_mind

from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, PrivateAttr, root_validator

from langchain_community.utilities.mindsdb.database_mind.database_models import validate_data_source_connection_args

DEFAULT_API_BASE = "https://llm.mdb.ai"
DEFAULT_MODEL = "gpt-3.5-turbo"


# TODO: Support openai < 1
class DatabaseMindWrapper(BaseModel):
    mindsdb_api_key: SecretStr = Field(default=None)
    mindsdb_api_base: Text = Field(default=DEFAULT_API_BASE)
    model: Text = Field(default=DEFAULT_MODEL)
    name: Text
    description: Text
    data_source_type: Text = Literal['postgres']
    data_source_connection_args: Dict

    _mind: Mind = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        # Validate that the `mindsdb_sdk` and `openai' packages can be imported.
        try:
            import openai

        except ImportError as e:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`.",
            ) from e

        try:
            import mindsdb_sdk

        except ImportError as e:
            raise ImportError(
                "Could not import mindsdb_sdk python package. "
                "Please install it with `pip install mindsdb_sdk`.",
            ) from e

        # Validate that the correct connection arguments are provided for the chosen data source.
        validate_data_source_connection_args(values['data_source_type'], values['data_source_connection_args'])

        return values

    def _create_mind(self) -> None:
        self._mind = create_mind(
            name=self.name,
            description=self.description,
            base_url=self.mindsdb_api_base,
            api_key=self.mindsdb_api_key.get_secret_value(),
            model=self.model,
            data_source_type=self.data_source_type,
            data_source_connection_args=self.data_source_connection_args
        )

    def _create_client(self) -> None:
        self._client = OpenAI(
            api_key=self.mindsdb_api_key.get_secret_value(),
            base_url=self.mindsdb_api_base
        )

    def _query_mind(self, query: Text) -> Text:
        completion = self._client.chat.completions.create(
            model=self._mind.name,
            messages=[
                {'role': 'user', 'content': query}
            ],
            stream=False
        )

        return completion.choices[0].message.content

    def run(self, query: Text) -> Text:
        if not self._mind:
            self._create_mind()

        if not self._client:
            self._create_client()

        return self._query_mind(query)
