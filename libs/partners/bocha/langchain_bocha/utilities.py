"""Wrapper for Bocha Search API."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import requests
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

logger = logging.getLogger(__name__)


class BochaAPIWrapper(BaseModel):
    """Wrapper for Bocha Search API.

    To use, you should have the environment variable ``BOCHA_API_KEY``
    set with your API key, or pass it as a named parameter to the
    constructor.
    """

    bocha_api_key: SecretStr | None = Field(default=None)
    endpoint: str = Field(default="https://api.bocha.cn/v1/web-search")

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> Any:
        """Validate that api key exists in environment or values."""
        bocha_api_key = get_from_dict_or_env(values, "bocha_api_key", "BOCHA_API_KEY")
        values["bocha_api_key"] = convert_to_secret_str(bocha_api_key)
        return values

    def _prepare_request(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
        summary: bool | None = None,
        include: str | None = None,
        exclude: str | None = None,
    ) -> dict[str, Any]:
        """Prepare the request payload and headers."""
        headers = {
            "Authorization": f"Bearer {self.bocha_api_key.get_secret_value()}",  # type: ignore[union-attr]
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {"query": query}
        if count is not None:
            payload["count"] = count
        if freshness is not None:
            payload["freshness"] = freshness
        if summary is not None:
            payload["summary"] = summary
        if include is not None:
            payload["include"] = include
        if exclude is not None:
            payload["exclude"] = exclude

        return {"headers": headers, "json": payload}

    def _parse_results(self, response_json: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse search results from API response json."""
        web_pages = response_json.get("webPages", {})
        values = web_pages.get("value", [])

        return [
            {
                "title": val.get("name", ""),
                "link": val.get("url", ""),
                "snippet": val.get("snippet", ""),
                "summary": val.get("summary", ""),
                "site_name": val.get("siteName", ""),
                "date_published": val.get("datePublished", ""),
            }
            for val in values
        ]

    def results(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
        summary: bool | None = None,
        include: str | None = None,
        exclude: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run query through Bocha Search API and return metadata.

        Args:
            query: The query to search for.
            count: Number of results to return. Default is 10.
            freshness: Search time range. Default is "noLimit".
            summary: Whether to return text summary. Default is False.
            include: Domains to include in search.
            exclude: Domains to exclude from search.

        Returns:
            A list of dictionaries with the search results.
        """
        request_params = self._prepare_request(
            query=query,
            count=count,
            freshness=freshness,
            summary=summary,
            include=include,
            exclude=exclude,
        )

        try:
            response = requests.post(
                self.endpoint,
                headers=request_params["headers"],
                json=request_params["json"],
                timeout=10,
            )
            response.raise_for_status()
            return self._parse_results(response.json())
        except Exception as e:
            msg = f"Error fetching results from Bocha API: {e}"
            raise ValueError(msg) from e

    async def aresults(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
        summary: bool | None = None,
        include: str | None = None,
        exclude: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run query through Bocha Search API and return metadata asynchronously.

        Args:
            query: The query to search for.
            count: Number of results to return. Default is 10.
            freshness: Search time range. Default is "noLimit".
            summary: Whether to return text summary. Default is False.
            include: Domains to include in search.
            exclude: Domains to exclude from search.

        Returns:
            A list of dictionaries with the search results.
        """
        request_params = self._prepare_request(
            query=query,
            count=count,
            freshness=freshness,
            summary=summary,
            include=include,
            exclude=exclude,
        )

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    self.endpoint,
                    headers=request_params["headers"],
                    json=request_params["json"],
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response,
            ):
                response.raise_for_status()
                response_json = await response.json()
                return self._parse_results(response_json)
        except Exception as e:
            msg = f"Error fetching results from Bocha API: {e}"
            raise ValueError(msg) from e

    def run(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
        summary: bool | None = None,
        include: str | None = None,
        exclude: str | None = None,
    ) -> str:
        """Run query through Bocha Search and parse result as a string.

        Args:
            query: The query to search for.
            count: Number of results to return. Default is 10.
            freshness: Search time range. Default is "noLimit".
            summary: Whether to return text summary. Default is False.
            include: Domains to include in search.
            exclude: Domains to exclude from search.

        Returns:
            A string formatted search results.
        """
        try:
            results = self.results(
                query=query,
                count=count,
                freshness=freshness,
                summary=summary,
                include=include,
                exclude=exclude,
            )
            return self._format_results(results)
        except Exception as e:
            return f"Error: {e}"

    async def arun(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
        summary: bool | None = None,
        include: str | None = None,
        exclude: str | None = None,
    ) -> str:
        """Run query through Bocha Search asynchronously and parse result as a string.

        Args:
            query: The query to search for.
            count: Number of results to return. Default is 10.
            freshness: Search time range. Default is "noLimit".
            summary: Whether to return text summary. Default is False.
            include: Domains to include in search.
            exclude: Domains to exclude from search.

        Returns:
            A string formatted search results.
        """
        try:
            results = await self.aresults(
                query=query,
                count=count,
                freshness=freshness,
                summary=summary,
                include=include,
                exclude=exclude,
            )
            return self._format_results(results)
        except Exception as e:
            return f"Error: {e}"

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """Format the search results into a string."""
        if not results:
            return "No good Bocha Search Result Was Found"

        formatted = []
        for res in results:
            content = res["summary"] if res.get("summary") else res["snippet"]
            formatted.append(
                f"Title: {res['title']}\nLink: {res['link']}\nSnippet: {content}\n"
            )
        return "\n".join(formatted)
