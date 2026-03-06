"""Tool for the Seltz Search API."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator
from seltz import Includes, Seltz  # type: ignore[import-untyped]

from langchain_seltz._utilities import initialize_client


class SeltzSearchResults(BaseTool):  # type: ignore[override]
    r"""Seltz Search tool.

    Setup:
        Install `langchain-seltz` and set environment variable `SELTZ_API_KEY`.

        ```bash
        pip install -U langchain-seltz
        export SELTZ_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_seltz import SeltzSearchResults

        tool = SeltzSearchResults()
        ```

    Invocation with args:
        ```python
        tool.invoke({"query": "what is the weather in SF", "max_documents": 5})
        ```

        ```python
        [{"url": "https://example.com/weather", "content": "Weather info..."}, ...]
        ```

    Invocation with ToolCall:

        ```python
        tool.invoke(
            {
                "args": {"query": "what is the weather in SF", "max_documents": 5},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```

        ```python
        ToolMessage(
            content='[{"url": "https://example.com", "content": "..."}]',
            name="seltz_search_results_json",
            tool_call_id="1",
        )
        ```
    """

    name: str = "seltz_search_results_json"
    description: str = (
        "A wrapper around Seltz Search. "
        "Input should be a search query. "
        "Output is a JSON array of the query results with url and content keys."
    )
    client: Seltz = Field(default=None)  # type: ignore[assignment]
    seltz_api_key: SecretStr = Field(default=SecretStr(""))
    seltz_endpoint: str | None = None
    seltz_insecure: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _run(
        self,
        query: str,
        max_documents: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict] | str:
        """Use the tool.

        Args:
            query: The search query.
            max_documents: The maximum number of documents to return. Default: 10
            run_manager: The run manager for callbacks.

        Returns:
            A list of dictionaries with url and content keys, or an error string.
        """
        try:
            response = self.client.search(
                query=query, includes=Includes(max_documents=max_documents)
            )
            return [
                {"url": doc.url, "content": doc.content} for doc in response.documents
            ]
        except Exception as e:
            return repr(e)
