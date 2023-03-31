"""Lightweight wrapper around requests library, with async support."""
from typing import Any, Dict, Optional

import aiohttp
import requests
from pydantic import BaseModel, Extra

REQUEST_RESPONSE_WITH_STATUS_CODES = "{status_code}: {reason}\n\n{text}"


def format_response(
    response: requests.Response,
    template: Optional[str] = None,
) -> str:
    """Format the response as text."""
    if template is None:
        return response.text
    else:
        response_fields = {
            "status_code": response.status,
            "headers": response.headers,
            "text": response.text,
            "reason": response.reason,
        }
        return template.format(**response_fields)


async def aformat_response(
    response: aiohttp.ClientResponse,
    template: Optional[str] = None,
) -> str:
    if isinstance(response, aiohttp.ClientResponse):
        text = await response.text()
    else:
        raise TypeError("Invalid response type.")
    if template is None:
        return text
    response_fields = {
        "status_code": response.status,
        "headers": response.headers,
        "text": text,
        "reason": response.reason,
    }
    return template.format(**response_fields)


class RequestsWrapper(BaseModel):
    """Lightweight wrapper around requests library."""

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    response_format_template: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text."""
        res = requests.get(url, headers=self.headers, **kwargs)
        return format_response(res, self.response_format_template)

    def post(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text."""
        res = requests.post(url, json=data, headers=self.headers, **kwargs)
        return format_response(res, self.response_format_template)

    def patch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text."""
        res = requests.patch(url, json=data, headers=self.headers, **kwargs)
        return format_response(res, self.response_format_template)

    def put(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text."""
        res = requests.put(url, json=data, headers=self.headers, **kwargs)
        return format_response(res, self.response_format_template)

    def delete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text."""
        res = requests.delete(url, headers=self.headers, **kwargs)
        return format_response(res, self.response_format_template)

    async def _arequest(self, method: str, url: str, **kwargs: Any) -> str:
        """Make an async request."""
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=self.headers, **kwargs
                ) as response:
                    return await aformat_response(
                        response, template=self.response_format_template
                    )
        else:
            async with self.aiosession.request(
                method, url, headers=self.headers, **kwargs
            ) as response:
                return await aformat_response(
                    response, template=self.response_format_template
                )

    async def aget(self, url: str, **kwargs: Any) -> str:
        """GET the URL and return the text asynchronously."""
        return await self._arequest("GET", url, **kwargs)

    async def apost(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """POST to the URL and return the text asynchronously."""
        return await self._arequest("POST", url, json=data, **kwargs)

    async def apatch(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PATCH the URL and return the text asynchronously."""
        return await self._arequest("PATCH", url, json=data, **kwargs)

    async def aput(self, url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        """PUT the URL and return the text asynchronously."""
        return await self._arequest("PUT", url, json=data, **kwargs)

    async def adelete(self, url: str, **kwargs: Any) -> str:
        """DELETE the URL and return the text asynchronously."""
        return await self._arequest("DELETE", url, **kwargs)
