"""Util that calls Merriam-Webster."""
import json
from typing import Dict, Optional
from urllib.parse import quote

import requests as req

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

MERRIAM_WEBSTER_API_URL = (
    "https://www.dictionaryapi.com/api/v3/references/collegiate/json"
)
MERRIAM_WEBSTER_TIMEOUT = 5000


class MerriamWebsterAPIWrapper(BaseModel):
    """Wrapper for Merriam-Webster.

    Docs for using:

    1. Go to https://www.dictionaryapi.com/register/index and register an
       developer account with a key for the Collegiate Dictionary
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

        request_url = (
            f"{MERRIAM_WEBSTER_API_URL}/{quoted_query}"
            f"?key={self.merriam_webster_api_key}"
        )

        response = req.get(request_url, timeout=MERRIAM_WEBSTER_TIMEOUT)

        if response.status_code != 200:
            return response.text

        return self._format_response(query, response)

    def _format_response(self, query: str, response: req.Response) -> str:
        content = json.loads(response.content)

        if not content:
            return f"No Merriam-Webster definition was found for query '{query}'."

        if isinstance(content[0], str):
            result = f"No Merriam-Webster definition was found for query '{query}'.\n"
            if len(content) > 1:
                alternatives = [f"{i + 1}. {content[i]}" for i in range(len(content))]
                result += "You can try one of the following alternative queries:\n\n"
                result += "\n".join(alternatives)
            else:
                result += f"Did you mean '{content[0]}'?"
        else:
            result = response.text  # TODO: Implement this

        return result
