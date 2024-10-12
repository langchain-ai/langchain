from typing import Optional

import requests
from pydantic import BaseModel

class GoogleBooksAPIWrapper(BaseModel):
    google_books_api_key: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        google_books_api_key = get_from_dict_or_env(
          values, "google_books_api_key", "GOOGLE_BOOKS_API_KEY"
        )
        values["google_books_api_key"] = google_books_api_key

        return values

    def run(self, query: str) -> str:
        # build Url based on API key and query
        request_url = ...

        # send request
        response = requests.get(request_url)

        # some error handeling
        if response.status_code != 200:
            return response.text

        # send back data (format style tbd)
        return self.format(query, response)