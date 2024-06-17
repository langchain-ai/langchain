from typing import Text, Dict, Set, Optional
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_community.chat_models.openai import (
    ChatOpenAI,
)

DEFAULT_API_BASE = "https://llm.mdb.ai"
DEFAULT_MODEL = "gpt-3.5-turbo"


class ChatMinds(ChatOpenAI):
    @property
    def _llm_type(self) -> Text:
        """Return type of chat model."""
        return "minds-chat"

    @property
    def lc_secrets(self) -> Dict[Text, Text]:
        return {"mindsdb_api_key": "MINDSDB_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    mindsdb_api_key: SecretStr = Field(default=None)

    model_name: str = Field(default=DEFAULT_MODEL, alias="model")

    anyscale_api_base: str = Field(default=DEFAULT_API_BASE)

    @staticmethod
    def get_available_models(
        anyscale_api_key: Optional[Text] = None,
        anyscale_api_base: str = DEFAULT_API_BASE,
    ) -> Set[Text]:
        pass

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        pass