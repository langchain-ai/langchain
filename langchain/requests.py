"""Lightweight wrapper around requests library, with async support."""
from typing import Any, Dict, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel, Extra


class RequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library."""

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    return_response: bool = False
    """Whether to return the response object directly."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _process_response(
        self, response: requests.Response
    ) -> Union[str, requests.Response]:
        """Return the appropriate result based on the return_response attribute."""
        if self.return_response:
            return response
        else:
            return response.text

    def get(self, url: str, **kwargs: Any) -> Union[str, requests.Response]:
        """GET the URL and return the text or the response object based on return_response."""
        response = requests.get(url, headers=self.headers, **kwargs)
        return self._process_response(response)

    def post(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, requests.Response]:
        """POST to the URL and return the text or the response object based on return_response."""
        response = requests.post(url, json=data, headers=self.headers, **kwargs)
        return self._process_response(response)

    def patch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, requests.Response]:
        """PATCH the URL and return the text or the response object based on return_response."""
        response = requests.patch(url, json=data, headers=self.headers, **kwargs)
        return self._process_response(response)

    def put(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[str, requests.Response]:
        """PUT the URL and return the text or the response object based on return_response."""
        response = requests.put(url, json=data, headers=self.headers, **kwargs)
        return self._process_response(response)

    def delete(self, url: str, **kwargs: Any) -> Union[str, requests.Response]:
        """DELETE the URL and return the text or the response object based on return_response."""
        response = requests.delete(url, headers=self.headers, **kwargs)
        return self._process_response(response)

    async def _arequest(
        self, method: str, url: str, **kwargs: Any
    ) -> Union[aiohttp.ClientResponse, str]:
        """Make an async request and return the appropriate result based on the return_response attribute."""

        async def _get_response():
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

        response = await _get_response()

        if self.return_response:
            return response
        else:
            return await response.text()

    async def aget(self, url: str, **kwargs: Any) -> Union[aiohttp.ClientResponse, str]:
        """GET the URL and return the text asynchronously."""
        return await self._arequest("GET", url, **kwargs)

    async def apost(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[aiohttp.ClientResponse, str]:
        """POST to the URL and return the text asynchronously."""
        return await self._arequest("POST", url, json=data, **kwargs)

    async def apatch(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[aiohttp.ClientResponse, str]:
        """PATCH the URL and return the text asynchronously."""
        return await self._arequest("PATCH", url, json=data, **kwargs)

    async def aput(
        self, url: str, data: Dict[str, Any], **kwargs: Any
    ) -> Union[aiohttp.ClientResponse, str]:
        """PUT the URL and return the text asynchronously."""
        return await self._arequest("PUT", url, json=data, **kwargs)

    async def adelete(
        self, url: str, **kwargs: Any
    ) -> Union[aiohttp.ClientResponse, str]:
        """DELETE the URL and return the text asynchronously."""
        return await self._arequest("DELETE", url, **kwargs)
