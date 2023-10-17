"""Util that calls Merriam-Webster."""
from typing import Dict, Optional

import requests as req
from urllib.parse import quote

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

MERRIAM_WEBSTER_API_URL = "https://www.dictionaryapi.com/api/v3/references/collegiate/json"
MERRIAM_WEBSTER_TIMEOUT = 5000

class MerriamWebsterAPIWrapper(BaseModel):
    """Wrapper for Merriam-Webster.

    Docs for using:

    1. Go to https://www.dictionaryapi.com/register/index and register an developer account with a key for the Collegiate Dictionary
    2. Get your API Key from https://www.dictionaryapi.com/account/my-keys
    3. Save your API Key into MERRIAM_WEBSTER_API_KEY env variable

    """

    merriam_webster_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        merriam_webster_api_key = get_from_dict_or_env(
            values, "merriam_webster_api_key", "MERRIAM_WEBSTER_API_KEY"
        )
        values["merriam_webster_api_key"] = merriam_webster_api_key

        return values

    def run(self, query: str) -> str:
        """Run query through Merriam-Webster API and return a formatted result."""
        quoted_query = quote(query)

        request_url = f"{MERRIAM_WEBSTER_API_URL}/{quoted_query}?key={self.merriam_webster_api_key}"

        response = req.get(request_url, timeout=MERRIAM_WEBSTER_TIMEOUT)
        return response.text
