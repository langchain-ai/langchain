"""Tool for the DuckDuckGo search API."""

import warnings
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


class DDGInput(BaseModel):
    """Input for the DuckDuckGo search tool."""

    query: str = Field(description="поисковый запрос")


class DuckDuckGoSearchRun(BaseTool):
    """Утилита для обращения к API поисковой системе DuckDuckGo"""

    name: str = "duckduckgo_search"
    description: str = (
        "Поиск в DuckDuckGo. "
        "Полезен, когда нужно ответить на вопросы о текущих событиях. "
        "Входными данными должен быть поисковый запрос."
    )
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )
    args_schema: Type[BaseModel] = DDGInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


class DuckDuckGoSearchResults(BaseTool):
    """Tool that queries the DuckDuckGo search API and gets back json."""

    name: str = "duckduckgo_results_json"
    description: str = (
        "Обертка вокруг поиска DuckDuckGo. "
        "Полезно, когда вам нужно ответить на вопросы о текущих событиях. "
        "Входными данными должен быть поисковый запрос. "
        "Выходом является JSON-массив результатов запроса."
    )
    max_results: int = Field(alias="num_results", default=4)
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )
    backend: str = "text"
    args_schema: Type[BaseModel] = DDGInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        res = self.api_wrapper.results(query, self.max_results, source=self.backend)
        res_strs = [", ".join([f"{k}: {v}" for k, v in d.items()]) for d in res]
        return ", ".join([f"[{rs}]" for rs in res_strs])


def DuckDuckGoSearchTool(*args: Any, **kwargs: Any) -> DuckDuckGoSearchRun:
    """
    Deprecated. Use DuckDuckGoSearchRun instead.

    Args:
        *args:
        **kwargs:

    Returns:
        DuckDuckGoSearchRun
    """
    warnings.warn(
        "DuckDuckGoSearchTool will be deprecated in the future. "
        "Please use DuckDuckGoSearchRun instead.",
        DeprecationWarning,
    )
    return DuckDuckGoSearchRun(*args, **kwargs)
