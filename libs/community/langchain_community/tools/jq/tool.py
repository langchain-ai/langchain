# flake8: noqa
"""Tools for working with JQ expression"""
from __future__ import annotations

import json
from typing import Optional, Literal

from langchain_core.utils import guard_import

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class JQ(BaseTool):
    """Tool to filter, transform, and format JSON data."""

    name: str = "jq_transform_doc"
    description: str = """
    Can be used to filter, transform, and format JSON data.
    The input should be a valid jq expression.
    """

    query_expression: str
    """JQ query expression"""

    return_type: Literal["text", "all", "first"] = "first"
    """return type from jq - text, all or first. 
    the default value is "first". 
    """

    def _impl(self, tool_input: str) -> str:
        jq = guard_import("jq")
        doc = json.loads(tool_input)

        query = jq.compile(self.query_expression)
        doc_next = query.input(doc)

        doc_next = getattr(doc_next, self.return_type)()
        if self.return_type != "text":
            text = json.dumps(doc_next)
        else:
            text = doc_next
        return text

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._impl(tool_input)

    async def _arun(
        self,
        tool_input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._impl(tool_input)
