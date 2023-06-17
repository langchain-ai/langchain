import json

import requests
from pydantic import BaseModel, Field


class BraveSearchWrapper(BaseModel):
    api_key: str
    search_kwargs: dict = Field(default_factory=dict)

    def run(self, query: str) -> str:
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        base_url = "https://api.search.brave.com/res/v1/web/search"
        req = requests.PreparedRequest()
        params = {**self.search_kwargs, **{"q": query}}
        req.prepare_url(base_url, params)
        if req.url is None:
            raise ValueError("prepared url is None, this should not happen")

        response = requests.get(req.url, headers=headers)

        if not response.ok:
            raise Exception(f"HTTP error {response.status_code}")

        parsed_response = response.json()
        web_search_results = parsed_response.get("web", {}).get("results", [])
        final_results = []
        if isinstance(web_search_results, list):
            for item in web_search_results:
                final_results.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("url"),
                        "snippet": item.get("description"),
                    }
                )
        return json.dumps(final_results)
