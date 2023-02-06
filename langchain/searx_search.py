"""Chain that calls SearxAPI.

This is developed based on the SearxNG fork https://github.com/searxng/searxng
For Searx API refer to https://docs.searxng.org/index.html
"""

import requests
from pydantic import BaseModel, PrivateAttr, Extra, Field, validator, root_validator
from typing import Optional, List, Dict, Any
import json


def _get_default_params() -> dict:
    return {
        # "engines": "google",
        "lang": "en",
        "format": "json"
    }


class SearxResults(dict):
    _data = ''

    def __init__(self, data: str):
        """
        Takes a raw result from Searx and make it into a dict like object
        """
        json_data = json.loads(data)
        super().__init__(json_data)
        self.__dict__ = self

    def __str__(self) -> str:
        return self._data

    # the following are fields from the json result of Searx we put getter
    # to silence mypy errors
    @property
    def results(self) -> Any:
        return self.results

    @property
    def answers(self) -> Any:
        return self.results


class SearxSearchWrapper(BaseModel):
    _result: SearxResults = PrivateAttr()
    host: str = ""
    unsecure: bool = False
    params: dict = Field(default_factory=_get_default_params)
    headers: Optional[dict] = None
    k: int = 10


    @validator("unsecure", pre=True)
    def disable_ssl_warnings(cls, v: bool) -> bool:
        if v:
            # requests.urllib3.disable_warnings()
            try:
                import urllib3
                urllib3.disable_warnings()
            except ImportError as e:
                print(e)

        return v

    @root_validator()
    def validate_params(cls, values: Dict) -> Dict:
        """Validate that custom searx params are merged with default ones"""
        user_params = values["params"]
        default = _get_default_params()
        values["params"] = {**default, **user_params}

        return values


    @validator("host", pre=True, always=True)
    def valid_host_url(cls, host: str) -> str:
        if len(host) == 0:
            raise ValueError("url can not be empty")
        if not host.startswith("http"):
            host = "http://" + host
        return host

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def _searx_api_query(self, params: dict) -> SearxResults:
        """actual request to searx API """
        raw_result = requests.get(self.host, headers=self.headers
                            , params=params,
                            verify=not self.unsecure).text
        self._result = SearxResults(raw_result)
        return self._result


    def run(self, query: str) -> str:
        """Run query through Searx API and parse results"""
        _params = { 
            "q": query,
       }
        params = {**self.params, **_params}
        res = self._searx_api_query(params)

        if len(res.answers) > 0:
            toret = res.answers[0]

        # only return the content of the results list
        elif len(res.results) > 0:
            toret = "\n\n".join([r['content'] for r in res.results[:self.k]])
        else:
            toret = "No good search result found"

        return toret

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through Searx API and returns the results with metadata.

            Args:
                query: The query to search for.
                num_results: Limit the number of results to return.

            Returns:
                A list of dictionaries with the following keys:
                    snippet - The description of the result.
                    title - The title of the result.
                    link - The link to the result.
        """
        metadata_results = []
        _params = {
                "q": query,
        }
        params = {**self.params, **_params}
        results = self._searx_api_query(params).results[:num_results]
        if len(results) == 0:
            return [{"Result": "No good Search Result was found"}]
        for result in results:
            metadata_result = {
                "snippet": result["content"],
                "title": result["title"],
                "link": result["url"],
            }
            metadata_results.append(metadata_result)

        return metadata_results


# if __name__ == "__main__":
#     search = SearxSearchWrapper(host='search.c.gopher', unsecure=True)
#     print(search.run("who is the current president of Bengladesh ?"))
