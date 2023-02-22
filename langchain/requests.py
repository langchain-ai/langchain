"""Lightweight wrapper around requests library, with async support."""
from typing import Any, Dict, Optional

import aiohttp
import requests
from pydantic import BaseModel, Extra


class RequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library."""

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get(self, url: str) -> str:
        """GET the URL and return the text."""
        return requests.get(url, headers=self.headers).text

    def post(self, url: str, data: Dict[str, Any]) -> str:
        """POST to the URL and return the text."""
        return requests.post(url, json=data, headers=self.headers).text

    def patch(self, url: str, data: Dict[str, Any]) -> str:
        """PATCH the URL and return the text."""
        return requests.patch(url, json=data, headers=self.headers).text

    def put(self, url: str, data: Dict[str, Any]) -> str:
        """PUT the URL and return the text."""
        return requests.put(url, json=data, headers=self.headers).text

    def delete(self, url: str) -> str:
        """DELETE the URL and return the text."""
        return requests.delete(url, headers=self.headers).text

    async def aget(self, url: str) -> str:
        """GET the URL and return the text asynchronously."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    return await response.text()
        else:
            async with self.aiosession.get(url, headers=self.headers) as response:
                return await response.text()

    async def apost(self, url: str, data: Dict[str, Any]) -> str:
        """POST to the URL and return the text asynchronously."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=data, headers=self.headers
                ) as response:
                    return await response.text()
        else:
            async with self.aiosession.post(
                url, json=data, headers=self.headers
            ) as response:
                return await response.text()

    async def apatch(self, url: str, data: Dict[str, Any]) -> str:
        """PATCH the URL and return the text asynchronously."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    url, json=data, headers=self.headers
                ) as response:
                    return await response.text()
        else:
            async with self.aiosession.patch(
                url, json=data, headers=self.headers
            ) as response:
                return await response.text()

    async def aput(self, url: str, data: Dict[str, Any]) -> str:
        """PUT the URL and return the text asynchronously."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    url, json=data, headers=self.headers
                ) as response:
                    return await response.text()
        else:
            async with self.aiosession.put(
                url, json=data, headers=self.headers
            ) as response:
                return await response.text()

    async def adelete(self, url: str) -> str:
        """DELETE the URL and return the text asynchronously."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=self.headers) as response:
                    return await response.text()
        else:
            async with self.aiosession.delete(url, headers=self.headers) as response:
                return await response.text()
