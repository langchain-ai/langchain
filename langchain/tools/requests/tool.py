# flake8: noqa
"""Tools for making requests to an API endpoint."""
import asyncio
import json
from typing import Any, Dict, Union
import aiohttp

from pydantic import BaseModel
import requests

from langchain.requests import RequestsWrapper
from langchain.tools.base import BaseTool


def _parse_input(text: str) -> Dict[str, Any]:
    """Parse the json string into a dict."""
    return json.loads(text)


def _get_text(response: Union[str, requests.Response]) -> str:
    """Return the text from the response."""
    if isinstance(response, str):
        return response
    else:
        return response.text


async def _get_text_async(response: Union[str, aiohttp.ClientResponse]) -> str:
    """Return the text from the response."""
    if isinstance(response, str):
        return response
    else:
        return await response.text


class BaseRequestsTool(BaseModel):
    """Base class for requests tools."""

    requests_wrapper: RequestsWrapper


class RequestsGetTool(BaseRequestsTool, BaseTool):
    """Tool for making a GET request to an API endpoint."""

    name = "requests_get"
    description = "A portal to the internet. Use this when you need to get specific content from a website. Input should be a  url (i.e. https://www.google.com). The output will be the text response of the GET request."

    def _run(self, url: str) -> str:
        """Run the tool."""
        return _get_text(self.requests_wrapper.get(url))

    async def _arun(self, url: str) -> str:
        """Run the tool asynchronously."""
        return await _get_text_async(self.requests_wrapper.aget(url))


class RequestsPostTool(BaseRequestsTool, BaseTool):
    """Tool for making a POST request to an API endpoint."""

    name = "requests_post"
    description = """Use this when you want to POST to a website.
    Input should be a json string with two keys: "url" and "data".
    The value of "url" should be a string, and the value of "data" should be a dictionary of 
    key-value pairs you want to POST to the url.
    Be careful to always use double quotes for strings in the json string
    The output will be the text response of the POST request.
    """

    def _run(self, text: str) -> str:
        """Run the tool."""
        try:
            data = _parse_input(text)
            return _get_text(self.requests_wrapper.post(data["url"], data["data"]))
        except Exception as e:
            return repr(e)

    async def _arun(self, text: str) -> str:
        """Run the tool asynchronously."""
        try:
            data = _parse_input(text)
            return await _get_text_async(
                self.requests_wrapper.apost(data["url"], data["data"])
            )
        except Exception as e:
            return repr(e)


class RequestsPatchTool(BaseRequestsTool, BaseTool):
    """Tool for making a PATCH request to an API endpoint."""

    name = "requests_patch"
    description = """Use this when you want to PATCH to a website.
    Input should be a json string with two keys: "url" and "data".
    The value of "url" should be a string, and the value of "data" should be a dictionary of 
    key-value pairs you want to PATCH to the url.
    Be careful to always use double quotes for strings in the json string
    The output will be the text response of the PATCH request.
    """

    def _run(self, text: str) -> str:
        """Run the tool."""
        try:
            data = _parse_input(text)
            return _get_text_async(
                _get_text(self.requests_wrapper.patch(data["url"], data["data"]))
            )
        except Exception as e:
            return repr(e)

    async def _arun(self, text: str) -> str:
        """Run the tool asynchronously."""
        try:
            data = _parse_input(text)
            return await _get_text_async(
                self.requests_wrapper.apatch(data["url"], data["data"])
            )
        except Exception as e:
            return repr(e)


class RequestsPutTool(BaseRequestsTool, BaseTool):
    """Tool for making a PUT request to an API endpoint."""

    name = "requests_put"
    description = """Use this when you want to PUT to a website.
    Input should be a json string with two keys: "url" and "data".
    The value of "url" should be a string, and the value of "data" should be a dictionary of 
    key-value pairs you want to PUT to the url.
    Be careful to always use double quotes for strings in the json string.
    The output will be the text response of the PUT request.
    """

    def _run(self, text: str) -> str:
        """Run the tool."""
        try:
            data = _parse_input(text)
            return _get_text(self.requests_wrapper.put(data["url"], data["data"]))
        except Exception as e:
            return repr(e)

    async def _arun(self, text: str) -> str:
        """Run the tool asynchronously."""
        try:
            data = _parse_input(text)
            return await _get_text_async(
                self.requests_wrapper.aput(data["url"], data["data"])
            )
        except Exception as e:
            return repr(e)


class RequestsDeleteTool(BaseRequestsTool, BaseTool):
    """Tool for making a DELETE request to an API endpoint."""

    name = "requests_delete"
    description = "A portal to the internet. Use this when you need to make a DELETE request to a URL. Input should be a specific url, and the output will be the text response of the DELETE request."

    def _run(self, url: str) -> str:
        """Run the tool."""
        return _get_text(self.requests_wrapper.delete(url))

    async def _arun(self, url: str) -> str:
        """Run the tool asynchronously."""
        return await _get_text_async(self.requests_wrapper.adelete(url))
