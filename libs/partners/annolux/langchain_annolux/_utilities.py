import json
import os
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class AnnoluxSearchAPIWrapper(BaseModel):
    """Wrapper for Annolux Curated Search API.

    To use, you must have the environment variable ``ANNOLUX_API_KEY`` set with your
    API key, or pass ``annolux_api_key`` as a named parameter to the constructor.
    """

    annolux_api_key: Optional[SecretStr] = Field(default=None)
    annolux_api_url: str = Field(default="https://api.annolux.com/v1/search")
    k: int = Field(default=10, description="Max results to return (1-10)")
    ranking: str = Field(default="default", description="Ranking algorithm (default or provider)")
    deduplicate: bool = Field(default=True, description="Collapse duplicate/near-duplicate results")

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment or passed in kwargs."""
        annolux_api_key = values.get("annolux_api_key") or os.environ.get("ANNOLUX_API_KEY")
        if not annolux_api_key:
            raise ValueError(
                "Did not find annolux_api_key, please add an environment variable"
                " `ANNOLUX_API_KEY` which contains it, or pass"
                " `annolux_api_key` as a named parameter."
            )
        if isinstance(annolux_api_key, str):
            values["annolux_api_key"] = SecretStr(annolux_api_key)
        return values

    def _headers(self) -> Dict[str, str]:
        key = self.annolux_api_key.get_secret_value() if self.annolux_api_key else ""
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "langchain-annolux/0.1.0",
        }

    def raw_results(
        self,
        query: str,
        limit: Optional[int] = None,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute query and return raw API response."""
        payload: Dict[str, Any] = {
            "query": query,
            "limit": limit or self.k,
            "ranking": self.ranking,
            "deduplicate": self.deduplicate,
        }
        if domains:
            payload["domains"] = domains

        response = requests.post(
            self.annolux_api_url,
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    async def raw_results_async(
        self,
        query: str,
        limit: Optional[int] = None,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async execute query and return raw API response."""
        payload: Dict[str, Any] = {
            "query": query,
            "limit": limit or self.k,
            "ranking": self.ranking,
            "deduplicate": self.deduplicate,
        }
        if domains:
            payload["domains"] = domains

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.annolux_api_url,
                headers=self._headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                return await response.json()

    def results(
        self,
        query: str,
        max_results: Optional[int] = None,
        domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run query and return list of result dictionaries."""
        raw = self.raw_results(query, limit=max_results, domains=domains)
        return raw.get("results", [])

    async def results_async(
        self,
        query: str,
        max_results: Optional[int] = None,
        domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Async run query and return list of result dictionaries."""
        raw = await self.raw_results_async(query, limit=max_results, domains=domains)
        return raw.get("results", [])

    def run(self, query: str) -> str:
        """Run query through Annolux Search and return clean text summary."""
        results = self.results(query)
        if not results:
            return "No relevant results found on Annolux curated corpus."
        snippets = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            fetched_at = r.get("fetched_at", "")
            snippets.append(
                f"Title: {title}\nURL: {url}\nSnapshot Date: {fetched_at}\nContent: {snippet}\n"
            )
        return "\n---\n".join(snippets)

    async def arun(self, query: str) -> str:
        """Async run query through Annolux Search and return clean text summary."""
        results = await self.results_async(query)
        if not results:
            return "No relevant results found on Annolux curated corpus."
        snippets = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            fetched_at = r.get("fetched_at", "")
            snippets.append(
                f"Title: {title}\nURL: {url}\nSnapshot Date: {fetched_at}\nContent: {snippet}\n"
            )
        return "\n---\n".join(snippets)
