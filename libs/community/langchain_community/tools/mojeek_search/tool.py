from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from langchain_community.utilities.mojeek_search import MojeekSearchAPIWrapper


class MojeekSearch(BaseTool):  # type: ignore[override]
    name: str = "mojeek_search"
    description: str = (
        "A wrapper around Mojeek Search. "
        "Useful for when you need to web search results. "
        "Input should be a search query."
    )
    api_wrapper: MojeekSearchAPIWrapper

    @classmethod
    def config(
        cls, api_key: str, search_kwargs: Optional[dict] = None, **kwargs: Any
    ) -> MojeekSearch:
        wrapper = MojeekSearchAPIWrapper(
            api_key=api_key, search_kwargs=search_kwargs or {}
        )
        return cls(api_wrapper=wrapper, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("MojeekSearch does not support async")
