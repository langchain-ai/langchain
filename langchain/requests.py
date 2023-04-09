"""Lightweight wrapper around requests library, with async support."""
from typing import Any, Dict, Optional

import aiohttp
import requests
from pydantic import BaseModel, Extra


class Requests(BaseModel):
    """Wrapper around requests to handle auth and async.

    The main purpose of this wrapper is to handle authentication (by saving
    headers) and enable easy async methods on the same base object.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """GET the URL and return the text."""
        return requests.get(url, headers=self.headers, **kwargs)

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """POST to the URL and return the text."""
        return requests.post(url, json=data, headers=self.headers, **kwargs)

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PATCH the URL and return the text."""
        return requests.patch(url, json=data, headers=self.headers, **kwargs)

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PUT the URL and return the text."""
        return requests.put(url, json=data, headers=self.headers, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """DELETE the URL and return the text."""
        return requests.delete(url, headers=self.headers, **kwargs)

    async def _arequest(
        self, method: str, url: str, **kwargs: Any
    ) -> aiohttp.ClientResponse:
        """Make an async request."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=self.headers, **kwargs
                ) as response:
                    return response
        else:
            async with self.aiosession.request(
                method, url, headers=self.headers, **kwargs
            ) as response:
                return response

    async def aget(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """GET the URL and return the text asynchronously."""
        return await self._arequest("GET", url, **kwargs)

    async def apost(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> aiohttp.ClientResponse:
        """POST to the URL and return the text asynchronously."""
        return await self._arequest("POST", url, json=data, **kwargs)

    async def apatch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> aiohttp.ClientResponse:
        """PATCH the URL and return the text asynchronously."""
        return await self._arequest("PATCH", url, json=data, **kwargs)

    async def aput(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> aiohttp.ClientResponse:
        """PUT the URL and return the text asynchronously."""
        return await self._arequest("PUT", url, json=data, **kwargs)

    async def adelete(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """DELETE the URL and return the text asynchronously."""
        return await self._arequest("DELETE", url, **kwargs)


class TextRequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library.

    The main purpose of this wrapper is to always return a text output.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def requests(self) -> Requests:
        return Requests(headers=self.headers, aiosession=self.aiosession)

    def get(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text."""
        return self.requests.get(url, **kwargs).text

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text."""
        return self.requests.post(url, data, **kwargs).text

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text."""
        return self.requests.patch(url, data, **kwargs).text

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text."""
        return self.requests.put(url, data, **kwargs).text

    def delete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text."""
        return self.requests.delete(url, **kwargs).text

    async def aget(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text asynchronously."""
        response = await self.requests.aget(url, **kwargs)
        return await response.text()

    async def apost(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text asynchronously."""
        response = await self.requests.apost(url, data, **kwargs)
        return await response.text()

    async def apatch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text asynchronously."""
        response = await self.requests.apatch(url, data, **kwargs)
        return await response.text()

    async def aput(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text asynchronously."""
        response = await self.requests.aput(url, data, **kwargs)
        return await response.text()

    async def adelete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text asynchronously."""
        response = await self.requests.adelete(url, **kwargs)
        return await response.text()


# For backwards compatibility
RequestsWrapper = TextRequestsWrapper
