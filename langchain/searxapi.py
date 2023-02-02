"""Chain that calls SearxAPI. """

import requests
from pydantic import BaseModel, Extra, Field, validator
from typing import Optional

def _get_default_params() -> dict:
    return {
        # "engines": "google",
        "lang": "en",
        "format": "json"
    }


class SearxAPIWrapper(BaseModel):
    host: str = ''
    unsecure: bool = False
    params: dict = Field(default_factory=_get_default_params)
    headers: Optional[dict] = None

    @validator('unsecure', pre=True)
    def disable_ssl_warnings(cls, v: bool) -> bool:
        if v:
            # requests.urllib3.disable_warnings()
            try:
                import urllib3
                urllib3.disable_warnings()
            except ImportError as e:
                print(e)

        return v

    @validator('host', pre=True, always=True)
    def valid_host_url(cls, host: str) -> str:
        if len(host) == 0:
            raise ValueError("url can not be empty")
        if not host.startswith("http"):
            host = "http://" + host
        return host

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    #TODO: return a dict instead of text
    def _searx_api_query(self, params: dict) -> dict:
        """actual request to searx API """
        # print(locals())
        # print(self.host)
        return requests.get(self.host, headers=self.headers
                            , params=params,
                            verify=not self.unsecure).text

    def run(self, query: str) -> str:
        """Run query through Searx API and parse result"""
        _params = { 
            "q": query,
       }
        params = {**self.params, **_params}

        res = self._searx_api_query(params)
        return res



if __name__ == "__main__":
    search = SearxAPIWrapper(host='search.c.gopher', unsecure=True)
    print(search.run("hello world"))
