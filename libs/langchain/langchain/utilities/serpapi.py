"""Chain that calls SerpApi.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
from typing import Any, Dict, Optional, Tuple, cast

import aiohttp

from langchain.pydantic_v1 import BaseModel, Extra, Field, SecretStr, root_validator
from langchain.utils import convert_to_secret_str, get_from_dict_or_env


class SerpAPIWrapper(BaseModel):
    """Wrapper around SerpApi.

    To use, you should have the ``google-search-results`` python package installed,
    and the environment variable ``SERPAPI_API_KEY`` set with your API key, or pass
    `serpapi_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.utilities import SerpAPIWrapper
            serpapi = SerpAPIWrapper()
    """

    search_engine: Any  #: :meta private:
    params: dict = Field(
        default={
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
    )
    serpapi_api_key: Optional[SecretStr] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["serpapi_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "serpapi_api_key", "SERPAPI_API_KEY")
        )
        try:
            from serpapi import SerpApiClient

            values["search_engine"] = SerpApiClient
        except ImportError:
            raise ValueError(
                "Could not import serpapi python package. "
                "Please install it with `pip install google-search-results`."
            )
        return values

    async def arun(self, query: str, **kwargs: Any) -> str:
        """Run query through SerpApi and parse result async."""
        return self._process_response(await self.aresults(query))

    def run(self, query: str, **kwargs: Any) -> str:
        """Run query through SerpApi and parse result."""
        return self._process_response(self.results(query))

    def results(self, query: str) -> dict:
        """Run query through SerpApi and return the raw result."""
        params = self.get_params(query)
        search = self.search_engine(params)
        res = search.get_dict()
        return res

    @staticmethod
    def get_user_agent() -> str:
        from langchain import __version__

        return f"langchain-py/{__version__}"

    async def aresults(self, query: str) -> dict:
        """Use aiohttp to run query through SerpApi and return the results async."""

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            params = self.get_params(query)
            params["output"] = "json"
            url = "https://serpapi.com/search"
            return url, params

        url, params = construct_url_and_params()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get(url, params=params) as response:
                res = await response.json()

        return res

    def get_params(self, query: str) -> Dict[str, str]:
        """Get parameters for SerpApi."""
        _params = {
            "api_key": cast(SecretStr, self.serpapi_api_key).get_secret_value(),
            "source": self.get_user_agent(),
            "q": query,
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from SerpApi."""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpApi: {res['error']}")
        if "answer_box_list" in res.keys():
            res["answer_box"] = res["answer_box_list"]
        if "answer_box" in res.keys():
            answer_box = res["answer_box"]
            if isinstance(answer_box, list):
                answer_box = answer_box[0]
            if "result" in answer_box.keys():
                return answer_box["result"]
            elif "answer" in answer_box.keys():
                return answer_box["answer"]
            elif "snippet" in answer_box.keys():
                return answer_box["snippet"]
            elif "snippet_highlighted_words" in answer_box.keys():
                return answer_box["snippet_highlighted_words"]
            else:
                answer = {}
                for key, value in answer_box.items():
                    if not isinstance(value, (list, dict)) and not (
                        isinstance(value, str) and value.startswith("http")
                    ):
                        answer[key] = value
                return str(answer)
        elif "events_results" in res.keys():
            return res["events_results"][:10]
        elif "sports_results" in res.keys():
            return res["sports_results"]
        elif "top_stories" in res.keys():
            return res["top_stories"]
        elif "news_results" in res.keys():
            return res["news_results"]
        elif "jobs_results" in res.keys() and "jobs" in res["jobs_results"].keys():
            return res["jobs_results"]["jobs"]
        elif (
            "shopping_results" in res.keys()
            and "title" in res["shopping_results"][0].keys()
        ):
            return res["shopping_results"][:3]
        elif "questions_and_answers" in res.keys():
            return res["questions_and_answers"]
        elif (
            "popular_destinations" in res.keys()
            and "destinations" in res["popular_destinations"].keys()
        ):
            return res["popular_destinations"]["destinations"]
        elif "top_sights" in res.keys() and "sights" in res["top_sights"].keys():
            return res["top_sights"]["sights"]
        elif (
            "images_results" in res.keys()
            and "thumbnail" in res["images_results"][0].keys()
        ):
            return str([item["thumbnail"] for item in res["images_results"][:10]])

        snippets = []
        if "knowledge_graph" in res.keys():
            knowledge_graph = res["knowledge_graph"]
            title = knowledge_graph["title"] if "title" in knowledge_graph else ""
            if "description" in knowledge_graph.keys():
                snippets.append(knowledge_graph["description"])
            for key, value in knowledge_graph.items():
                if (
                    isinstance(key, str)
                    and isinstance(value, str)
                    and key not in ["title", "description"]
                    and not key.endswith("_stick")
                    and not key.endswith("_link")
                    and not value.startswith("http")
                ):
                    snippets.append(f"{title} {key}: {value}.")

        for organic_result in res.get("organic_results", []):
            if "snippet" in organic_result.keys():
                snippets.append(organic_result["snippet"])
            elif "snippet_highlighted_words" in organic_result.keys():
                snippets.append(organic_result["snippet_highlighted_words"])
            elif "rich_snippet" in organic_result.keys():
                snippets.append(organic_result["rich_snippet"])
            elif "rich_snippet_table" in organic_result.keys():
                snippets.append(organic_result["rich_snippet_table"])
            elif "link" in organic_result.keys():
                snippets.append(organic_result["link"])

        if "buying_guide" in res.keys():
            snippets.append(res["buying_guide"])
        if "local_results" in res.keys() and "places" in res["local_results"].keys():
            snippets.append(res["local_results"]["places"])

        if len(snippets) > 0:
            return str(snippets)
        else:
            return "No good search result found"
