"""Util that calls Google Search using the Serper.dev API."""
from typing import Dict, Optional

import requests
from pydantic.class_validators import root_validator
from pydantic.main import BaseModel

from langchain.utils import get_from_dict_or_env


class GoogleSerperAPIWrapper(BaseModel):
    """Wrapper around the Serper.dev Google Search API.

    You can create a free API key at https://serper.dev.

    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor.

    Example:
        .. code-block:: python

            from langchain import GoogleSerperAPIWrapper
            google_serper = GoogleSerperAPIWrapper()
    """

    k: int = 10
    gl: str = "us"
    hl: str = "en"
    serper_api_key: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        serper_api_key = get_from_dict_or_env(
            values, "serper_api_key", "SERPER_API_KEY"
        )
        values["serper_api_key"] = serper_api_key

        return values

    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        results = self._google_serper_search_results(query, gl=self.gl, hl=self.hl)

        return self._parse_results(results)

    def _parse_results(self, results: dict) -> str:
        snippets = []

        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                return answer_box.get("answer")
            elif answer_box.get("snippet"):
                return answer_box.get("snippet").replace("\n", " ")
            elif answer_box.get("snippetHighlighted"):
                return ", ".join(answer_box.get("snippetHighlighted"))

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                snippets.append(f"{title}: {entity_type}.")
            description = kg.get("description")
            if description:
                snippets.append(description)
            for attribute, value in kg.get("attributes", {}).items():
                snippets.append(f"{title} {attribute}: {value}.")

        for result in results["organic"][: self.k]:
            if "snippet" in result:
                snippets.append(result["snippet"])
            for attribute, value in result.get("attributes", {}).items():
                snippets.append(f"{attribute}: {value}.")

        if len(snippets) == 0:
            return "No good Google Search Result was found"

        return " ".join(snippets)

    def _google_serper_search_results(self, search_term: str, gl: str, hl: str) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {"q": search_term, "gl": gl, "hl": hl}
        response = requests.post(
            "https://google.serper.dev/search", headers=headers, params=params
        )
        response.raise_for_status()
        search_results = response.json()
        return search_results
