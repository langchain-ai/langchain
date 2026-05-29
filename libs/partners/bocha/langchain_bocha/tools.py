"""Tools for the Bocha Search API."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator

from langchain_bocha._utils import (
    format_search_results,
    initialize_client,
    parse_search_results,
)


class BochaSearchRun(BaseTool):  # type: ignore[override]
    """Tool that queries the Bocha Search API.

    Returns a concatenated string of results.

    Setup:
        Install `langchain-bocha` and set environment variable `BOCHA_API_KEY`.

        .. code-block:: bash

            pip install -U langchain-bocha
            export BOCHA_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_bocha import BochaSearchRun

            tool = BochaSearchRun()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query": "what is the weather in Beijing"})
    """

    name: str = "bocha_search"
    description: str = (
        "A wrapper around Bocha Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    client: Any = Field(default=None, exclude=True)
    bocha_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool."""
        try:
            raw = self.client.post("/web-search", {"query": query, "count": 10})
            results = parse_search_results(raw)
            return format_search_results(results)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            raw = await self.client.apost("/web-search", {"query": query, "count": 10})
            results = parse_search_results(raw)
            return format_search_results(results)
        except Exception as e:
            return f"Error: {e}"


class BochaSearchResults(BaseTool):  # type: ignore[override]
    """Tool that queries the Bocha Search API and returns a JSON list of results.

    Setup:
        Install `langchain-bocha` and set environment variable `BOCHA_API_KEY`.

        .. code-block:: bash

            pip install -U langchain-bocha
            export BOCHA_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_bocha import BochaSearchResults

            tool = BochaSearchResults()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query": "what is the weather in Beijing"})
    """

    name: str = "bocha_search_results_json"
    description: str = (
        "A wrapper around Bocha Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. "
        "Output is a JSON array of the query results"
    )
    client: Any = Field(default=None, exclude=True)
    bocha_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Use the tool."""
        try:
            raw = self.client.post("/web-search", {"query": query, "count": 10})
            return parse_search_results(raw)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]] | str:
        """Use the tool asynchronously."""
        try:
            raw = await self.client.apost("/web-search", {"query": query, "count": 10})
            return parse_search_results(raw)
        except Exception as e:
            return f"Error: {e}"
