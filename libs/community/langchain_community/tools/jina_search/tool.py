from __future__ import annotations

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.jina_search import JinaSearchAPIWrapper


class JinaInput(BaseModel):
    """Input for the Jina search tool."""

    query: str = Field(description="search query to look up")


class JinaSearch(BaseTool):  # type: ignore[override]
    """Tool that queries the JinaSearch.

    ..versionadded:: 0.2.16
    """

    name: str = "jina_search"
    description: str = (
        "Jina Reader allows you to ground your LLM with the latest information from "
        "the web. "
        "Jina Reader will search the web and return the top five results with their "
        "URLs and contents, "
        "each in clean, LLM-friendly text. This way, you can always keep your LLM "
        "up-to-date, improve its factuality, and reduce hallucinations."
    )
    search_wrapper: JinaSearchAPIWrapper = Field(default_factory=JinaSearchAPIWrapper)  # type: ignore[arg-type]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.search_wrapper.run(query)
