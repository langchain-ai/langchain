from typing import Text
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr

DEFAULT_API_BASE = "https://llm.mdb.ai"
DEFAULT_MODEL = "gpt-3.5-turbo"


class BaseMindWrapper(BaseModel):
    mindsdb_api_key: SecretStr = Field(default=None)
    mindsdb_api_base: Text = Field(default=DEFAULT_API_BASE)
    model: Text = Field(default=DEFAULT_MODEL)
    name: Text
