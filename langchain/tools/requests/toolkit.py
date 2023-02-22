from typing import List

from langchain.requests import RequestsWrapper
from langchain.tools import BaseTool
from langchain.tools.base import BaseToolkit
from langchain.tools.requests.tool import (
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
)


class RequestsToolkit(BaseToolkit):
    """Toolkit for making requests."""

    requests_wrapper: RequestsWrapper

    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools."""
        return [
            RequestsGetTool(requests_wrapper=self.requests_wrapper),
            RequestsPostTool(requests_wrapper=self.requests_wrapper),
            RequestsPatchTool(requests_wrapper=self.requests_wrapper),
            RequestsPutTool(requests_wrapper=self.requests_wrapper),
            RequestsDeleteTool(requests_wrapper=self.requests_wrapper),
        ]
