# flake8: noqa
"""Tools for making requests to an API endpoint."""

import json
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain_community.utilities.requests import GenericRequestsWrapper
from langchain_core.tools import BaseTool


def _parse_input(text: str) -> Dict[str, Any]:
    """Parse the json string into a dict."""
    return json.loads(text)


def _clean_url(url: str) -> str:
    """Strips quotes from the url."""
    return url.strip("\"'")


class BaseRequestsTool(BaseModel):
    """Base class for requests tools."""

    requests_wrapper: GenericRequestsWrapper

    allow_dangerous_requests: bool = False

    def __init__(self, **kwargs: Any):
        """Initialize the tool."""
        if not kwargs.get("allow_dangerous_requests", False):
            raise ValueError(
                "You must set allow_dangerous_requests to True to use this tool. "
                "Requests can be dangerous and can lead to security vulnerabilities. "
                "For example, users can ask a server to make a request to an internal "
                "server. It's recommended to use requests through a proxy server "
                "and avoid accepting inputs from untrusted sources without proper "
                "sandboxing."
                "Please see: https://python.langchain.com/docs/security for "
                "further security information."
            )
        super().__init__(**kwargs)


class RequestsGetTool(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Tool for making a GET request to an API endpoint."""

    name: str = "requests_get"
    description: str = """A portal to the internet. Use this when you need to get specific
    content from a website. Input should be a  url (i.e. https://www.google.com).
    The output will be the text response of the GET request.
    """

    def _run(
        self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool."""
        return self.requests_wrapper.get(_clean_url(url))

    async def _arun(
        self,
        url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool asynchronously."""
        return await self.requests_wrapper.aget(_clean_url(url))


class RequestsPostTool(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Tool for making a POST request to an API endpoint."""

    name: str = "requests_post"
    description: str = """Use this when you want to POST to a website.
    Input should be a json string with two keys: "url" and "data".
    The value of "url" should be a string, and the value of "data" should be a dictionary of 
    key-value pairs you want to POST to the url.
    Be careful to always use double quotes for strings in the json string
    The output will be the text response of the POST request.
    """

    def _run(
        self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool."""
        try:
            data = _parse_input(text)
            return self.requests_wrapper.post(_clean_url(data["url"]), data["data"])
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        text: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool asynchronously."""
        try:
            data = _parse_input(text)
            return await self.requests_wrapper.apost(
                _clean_url(data["url"]), data["data"]
            )
        except Exception as e:
            return repr(e)


class RequestsPatchTool(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Tool for making a PATCH request to an API endpoint."""

    name: str = "requests_patch"
    description: str = """Use this when you want to PATCH to a website.
    Input should be a json string with two keys: "url" and "data".
    The value of "url" should be a string, and the value of "data" should be a dictionary of 
    key-value pairs you want to PATCH to the url.
    Be careful to always use double quotes for strings in the json string
    The output will be the text response of the PATCH request.
    """

    def _run(
        self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool."""
        try:
            data = _parse_input(text)
            return self.requests_wrapper.patch(_clean_url(data["url"]), data["data"])
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        text: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool asynchronously."""
        try:
            data = _parse_input(text)
            return await self.requests_wrapper.apatch(
                _clean_url(data["url"]), data["data"]
            )
        except Exception as e:
            return repr(e)


class RequestsPutTool(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Tool for making a PUT request to an API endpoint."""

    name: str = "requests_put"
    description: str = """Use this when you want to PUT to a website.
    Input should be a json string with two keys: "url" and "data".
    The value of "url" should be a string, and the value of "data" should be a dictionary of 
    key-value pairs you want to PUT to the url.
    Be careful to always use double quotes for strings in the json string.
    The output will be the text response of the PUT request.
    """

    def _run(
        self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool."""
        try:
            data = _parse_input(text)
            return self.requests_wrapper.put(_clean_url(data["url"]), data["data"])
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        text: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool asynchronously."""
        try:
            data = _parse_input(text)
            return await self.requests_wrapper.aput(
                _clean_url(data["url"]), data["data"]
            )
        except Exception as e:
            return repr(e)


class RequestsDeleteTool(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Tool for making a DELETE request to an API endpoint."""

    name: str = "requests_delete"
    description: str = """A portal to the internet.
    Use this when you need to make a DELETE request to a URL.
    Input should be a specific url, and the output will be the text
    response of the DELETE request.
    """

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool."""
        return self.requests_wrapper.delete(_clean_url(url))

    async def _arun(
        self,
        url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Run the tool asynchronously."""
        return await self.requests_wrapper.adelete(_clean_url(url))
