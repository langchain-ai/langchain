"""Lightweight wrapper around requests library, with async support."""
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Union

import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra
from requests import Response


class Requests(BaseModel):
    """Wrapper around requests to handle auth and async.

    The main purpose of this wrapper is to handle authentication (by saving
    headers) and enable easy async methods on the same base object.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    auth: Optional[Any] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """GET the URL and return the text."""
        return requests.get(url, headers=self.headers, auth=self.auth, **kwargs)

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """POST to the URL and return the text."""
        return requests.post(
            url, json=data, headers=self.headers, auth=self.auth, **kwargs
        )

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PATCH the URL and return the text."""
        return requests.patch(
            url, json=data, headers=self.headers, auth=self.auth, **kwargs
        )

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> requests.Response:
        """PUT the URL and return the text."""
        return requests.put(
            url, json=data, headers=self.headers, auth=self.auth, **kwargs
        )

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """DELETE the URL and return the text."""
        return requests.delete(url, headers=self.headers, auth=self.auth, **kwargs)

    @asynccontextmanager
    async def _arequest(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """Make an async request."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=self.headers, auth=self.auth, **kwargs
                ) as response:
                    yield response
        else:
            async with self.aiosession.request(
                method, url, headers=self.headers, auth=self.auth, **kwargs
            ) as response:
                yield response

    @asynccontextmanager
    async def aget(
        self, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """GET the URL and return the text asynchronously."""
        async with self._arequest("GET", url, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def apost(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """POST to the URL and return the text asynchronously."""
        async with self._arequest("POST", url, json=data, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def apatch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """PATCH the URL and return the text asynchronously."""
        async with self._arequest("PATCH", url, json=data, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def aput(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """PUT the URL and return the text asynchronously."""
        async with self._arequest("PUT", url, json=data, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def adelete(
        self, url: str, **kwargs: Any
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """DELETE the URL and return the text asynchronously."""
        async with self._arequest("DELETE", url, **kwargs) as response:
            yield response


class GenericRequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library."""

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    auth: Optional[Any] = None
    response_content_type: Literal["text", "json"] = "text"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def requests(self) -> Requests:
        return Requests(
            headers=self.headers, aiosession=self.aiosession, auth=self.auth
        )

    def _get_resp_content(self, response: Response) -> Union[str, Dict[str, Any]]:
        if self.response_content_type == "text":
            return response.text
        elif self.response_content_type == "json":
            return response.json()
        else:
            raise ValueError(f"Invalid return type: {self.response_content_type}")

    async def _aget_resp_content(
        self, response: aiohttp.ClientResponse
    ) -> Union[str, Dict[str, Any]]:
        if self.response_content_type == "text":
            return await response.text()
        elif self.response_content_type == "json":
            return await response.json()
        else:
            raise ValueError(f"Invalid return type: {self.response_content_type}")

    def get(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """GET the URL and return the text."""
        return self._get_resp_content(self.requests.get(url, **kwargs))

    def post(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """POST to the URL and return the text."""
        return self._get_resp_content(self.requests.post(url, data, **kwargs))

    def patch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """PATCH the URL and return the text."""
        return self._get_resp_content(self.requests.patch(url, data, **kwargs))

    def put(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """PUT the URL and return the text."""
        return self._get_resp_content(self.requests.put(url, data, **kwargs))

    def delete(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """DELETE the URL and return the text."""
        return self._get_resp_content(self.requests.delete(url, **kwargs))

    async def aget(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """GET the URL and return the text asynchronously."""
        async with self.requests.aget(url, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def apost(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """POST to the URL and return the text asynchronously."""
        async with self.requests.apost(url, data, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def apatch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """PATCH the URL and return the text asynchronously."""
        async with self.requests.apatch(url, data, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def aput(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """PUT the URL and return the text asynchronously."""
        async with self.requests.aput(url, data, **kwargs) as response:
            return await self._aget_resp_content(response)

    async def adelete(self, url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """DELETE the URL and return the text asynchronously."""
        async with self.requests.adelete(url, **kwargs) as response:
            return await self._aget_resp_content(response)


class JsonRequestsWrapper(GenericRequestsWrapper):
    """Lightweight wrapper around requests library, with async support.

    The main purpose of this wrapper is to always return a json output."""

    response_content_type: Literal["text", "json"] = "json"


class TextRequestsWrapper(GenericRequestsWrapper):
    """Lightweight wrapper around requests library, with async support.

    The main purpose of this wrapper is to always return a text output."""

    response_content_type: Literal["text", "json"] = "text"


# For backwards compatibility
RequestsWrapper = TextRequestsWrapper
