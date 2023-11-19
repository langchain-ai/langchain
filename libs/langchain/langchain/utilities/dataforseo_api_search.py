import base64
from typing import Dict, Optional
from urllib.parse import quote

import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env


class DataForSeoAPIWrapper(BaseModel):
    """Wrapper around the DataForSeo API."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    default_params: dict = Field(
        default={
            "location_name": "United States",
            "language_code": "en",
            "depth": 10,
            "se_name": "google",
            "se_type": "organic",
        }
    )
    """Default parameters to use for the DataForSEO SERP API."""
    params: dict = Field(default={})
    """Additional parameters to pass to the DataForSEO SERP API."""
    api_login: Optional[str] = None
    """The API login to use for the DataForSEO SERP API."""
    api_password: Optional[str] = None
    """The API password to use for the DataForSEO SERP API."""
    json_result_types: Optional[list] = None
    """The JSON result types."""
    json_result_fields: Optional[list] = None
    """The JSON result fields."""
    top_count: Optional[int] = None
    """The number of top results to return."""
    aiosession: Optional[aiohttp.ClientSession] = None
    """The aiohttp session to use for the DataForSEO SERP API."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that login and password exists in environment."""
        login = get_from_dict_or_env(values, "api_login", "DATAFORSEO_LOGIN")
        password = get_from_dict_or_env(values, "api_password", "DATAFORSEO_PASSWORD")
        values["api_login"] = login
        values["api_password"] = password
        return values

    async def arun(self, url: str) -> str:
        """Run request to DataForSEO SERP API and parse result async."""
        return self._process_response(await self._aresponse_json(url))

    def run(self, url: str) -> str:
        """Run request to DataForSEO SERP API and parse result async."""
        return self._process_response(self._response_json(url))

    def results(self, url: str) -> list:
        res = self._response_json(url)
        return self._filter_results(res)

    async def aresults(self, url: str) -> list:
        res = await self._aresponse_json(url)
        return self._filter_results(res)

    def _prepare_request(self, keyword: str) -> dict:
        """Prepare the request details for the DataForSEO SERP API."""
        if self.api_login is None or self.api_password is None:
            raise ValueError("api_login or api_password is not provided")
        cred = base64.b64encode(
            f"{self.api_login}:{self.api_password}".encode("utf-8")
        ).decode("utf-8")
        headers = {"Authorization": f"Basic {cred}", "Content-Type": "application/json"}
        obj = {"keyword": quote(keyword)}
        obj = {**obj, **self.default_params, **self.params}
        data = [obj]
        _url = (
            f"https://api.dataforseo.com/v3/serp/{obj['se_name']}"
            f"/{obj['se_type']}/live/advanced"
        )
        return {
            "url": _url,
            "headers": headers,
            "data": data,
        }

    def _check_response(self, response: dict) -> dict:
        """Check the response from the DataForSEO SERP API for errors."""
        if response.get("status_code") != 20000:
            raise ValueError(
                f"Got error from DataForSEO SERP API: {response.get('status_message')}"
            )
        return response

    def _response_json(self, url: str) -> dict:
        """Use requests to run request to DataForSEO SERP API and return results."""
        request_details = self._prepare_request(url)
        response = requests.post(
            request_details["url"],
            headers=request_details["headers"],
            json=request_details["data"],
        )
        response.raise_for_status()
        return self._check_response(response.json())

    async def _aresponse_json(self, url: str) -> dict:
        """Use aiohttp to request DataForSEO SERP API and return results async."""
        request_details = self._prepare_request(url)
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    request_details["url"],
                    headers=request_details["headers"],
                    json=request_details["data"],
                ) as response:
                    res = await response.json()
        else:
            async with self.aiosession.post(
                request_details["url"],
                headers=request_details["headers"],
                json=request_details["data"],
            ) as response:
                res = await response.json()
        return self._check_response(res)

    def _filter_results(self, res: dict) -> list:
        output = []
        types = self.json_result_types if self.json_result_types is not None else []
        for task in res.get("tasks", []):
            for result in task.get("result", []):
                for item in result.get("items", []):
                    if len(types) == 0 or item.get("type", "") in types:
                        self._cleanup_unnecessary_items(item)
                        if len(item) != 0:
                            output.append(item)
                    if self.top_count is not None and len(output) >= self.top_count:
                        break
        return output

    def _cleanup_unnecessary_items(self, d: dict) -> dict:
        fields = self.json_result_fields if self.json_result_fields is not None else []
        if len(fields) > 0:
            for k, v in list(d.items()):
                if isinstance(v, dict):
                    self._cleanup_unnecessary_items(v)
                    if len(v) == 0:
                        del d[k]
                elif k not in fields:
                    del d[k]

        if "xpath" in d:
            del d["xpath"]
        if "position" in d:
            del d["position"]
        if "rectangle" in d:
            del d["rectangle"]
        for k, v in list(d.items()):
            if isinstance(v, dict):
                self._cleanup_unnecessary_items(v)
        return d

    def _process_response(self, res: dict) -> str:
        """Process response from DataForSEO SERP API."""
        toret = "No good search result found"
        for task in res.get("tasks", []):
            for result in task.get("result", []):
                item_types = result.get("item_types")
                items = result.get("items", [])
                if "answer_box" in item_types:
                    toret = next(
                        item for item in items if item.get("type") == "answer_box"
                    ).get("text")
                elif "knowledge_graph" in item_types:
                    toret = next(
                        item for item in items if item.get("type") == "knowledge_graph"
                    ).get("description")
                elif "featured_snippet" in item_types:
                    toret = next(
                        item for item in items if item.get("type") == "featured_snippet"
                    ).get("description")
                elif "shopping" in item_types:
                    toret = next(
                        item for item in items if item.get("type") == "shopping"
                    ).get("price")
                elif "organic" in item_types:
                    toret = next(
                        item for item in items if item.get("type") == "organic"
                    ).get("description")
                if toret:
                    break
        return toret
