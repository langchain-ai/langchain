"""Util that calls Google Books."""

from typing import Dict, List, Optional

import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator

GOOGLE_BOOKS_MAX_ITEM_SIZE = 5
GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes"


class GoogleBooksAPIWrapper(BaseModel):
    """Wrapper around Google Books API.

    To use, you should have a Google Books API key available.
    This wrapper will use the Google Books API to conduct searches and
    fetch books based on a query passed in by the agents. By default,
    it will return the top-k results.

    The response for each book will contain the book title, author name, summary, and
    a source link.
    """

    google_books_api_key: Optional[str] = None
    top_k_results: int = GOOGLE_BOOKS_MAX_ITEM_SIZE

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        google_books_api_key = get_from_dict_or_env(
            values, "google_books_api_key", "GOOGLE_BOOKS_API_KEY"
        )
        values["google_books_api_key"] = google_books_api_key

        return values

    def run(self, query: str) -> str:
        # build Url based on API key, query, and max results
        params = (
            ("q", query),
            ("maxResults", self.top_k_results),
            ("key", self.google_books_api_key),
        )

        # send request
        response = requests.get(GOOGLE_BOOKS_API_URL, params=params)
        json = response.json()

        # some error handeling
        if response.status_code != 200:
            code = response.status_code
            error = json.get("error", {}).get("message", "Internal failure")
            return f"Unable to retrieve books got status code {code}: {error}"

        # send back data
        return self._format(query, json.get("items", []))

    def _format(self, query: str, books: List) -> str:
        if not books:
            return f"Sorry no books could be found for your query: {query}"

        start = f"Here are {len(books)} suggestions for books related to {query}:"

        results = []
        results.append(start)
        i = 1

        for book in books:
            info = book["volumeInfo"]
            title = info["title"]
            authors = self._format_authors(info["authors"])
            summary = info["description"]
            source = info["infoLink"]

            desc = f'{i}. "{title}" by {authors}: {summary}\n'
            desc += f"You can read more at {source}"
            results.append(desc)

            i += 1

        return "\n\n".join(results)

    def _format_authors(self, authors: List) -> str:
        if len(authors) == 1:
            return authors[0]
        return "{} and {}".format(", ".join(authors[:-1]), authors[-1])
