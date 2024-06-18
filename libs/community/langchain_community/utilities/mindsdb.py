from openai import OpenAI
from typing import Text, Dict, Literal
from mindsdb_sdk.utils.mind import Mind, create_mind

from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, PrivateAttr, root_validator

DEFAULT_API_BASE = "https://llm.mdb.ai"
DEFAULT_MODEL = "gpt-3.5-turbo"


class PostgresConnection(BaseModel):
    user: Text
    password: Text
    host: Text
    port: Text = Field(default=5432)
    database: Text
    schema: Text


class DatabaseMindWrapper(BaseModel):
    mindsdb_api_key: SecretStr = Field(default=None)
    mindsdb_api_base: Text = Field(default=DEFAULT_API_BASE)
    model: Text = Field(default=DEFAULT_MODEL)
    name: Text
    description: Text
    data_source_type: Text = Literal['postgres']
    data_source_connection_args: Dict

    _mind: Mind = PrivateAttr()

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
        if values['data_source_type'] == 'postgres':
            PostgresConnection(**values['data_source_connection_args'])

        return values

    def _create_mind(self) -> Mind:
        self._mind = create_mind(
            name=self.name,
            description=self.description,
            base_url=self.mindsdb_api_base,
            api_key=self.mindsdb_api_key,
            model=self.model,
            data_source_type=self.data_source_type,
            data_source_connection_args=self.data_source_connection_args
        )

    def _query_mind(self, query: Text) -> Text:
        pass

    def run(self, query: Text) -> Text:
        pass
