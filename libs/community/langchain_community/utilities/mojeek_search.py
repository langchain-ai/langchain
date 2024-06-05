import json
from typing import List

import requests
from langchain_core.pydantic_v1 import BaseModel, Field


class MojeekSearchAPIWrapper(BaseModel):
    api_key: str
    search_kwargs: dict = Field(default_factory=dict)
    api_url = "https://api.mojeek.com/search"

    def run(self, query: str) -> str:
        search_results = self._search(query)

        results = []

        for result in search_results:
            title = result.get("title", "")
            url = result.get("url", "")
            desc = result.get("desc", "")
            results.append({"title": title, "url": url, "desc": desc})

        return json.dumps(results)

    def _search(self, query: str) -> List[dict]:
        headers = {
            "Accept": "application/json",
        }

        req = requests.PreparedRequest()
        request = {
            **self.search_kwargs,
            **{"q": query, "fmt": "json", "api_key": self.api_key},
        }
        req.prepare_url(self.api_url, request)
        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")

        response = requests.get(req.url, headers=headers)
        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        return response.json().get("response", {}).get("results", [])
