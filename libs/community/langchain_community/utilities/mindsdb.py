from openai import OpenAI
from typing import Text, Dict, Optional
from mindsdb_sdk.utils.mind import Mind, create_mind

from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, PrivateAttr, root_validator

DEFAULT_API_BASE = "https://llm.mdb.ai"
DEFAULT_MODEL = "gpt-3.5-turbo"


class DatabaseMindWrapper(BaseModel):
    mindsdb_api_key: SecretStr = Field(default=None)
    mindsdb_api_base: Text = Field(default=DEFAULT_API_BASE)
    model: Text = Field(default=DEFAULT_MODEL)
    name: Text
    description: Text
    data_source_type: Text
    data_source_connection_args: Dict

    _mind: Mind = PrivateAttr()

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        pass

    def _create_mind(self) -> Mind:
        pass

    def _query_mind(self) -> Text:
        pass

    def run(self, query: Text) -> Text:
        pass
