from typing import Iterator, Optional

import httpx
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

QUERY_DATA_ENDPOINT = "https://api.agentql.com/v1/query-data"
API_TIMEOUT_SECONDS = 900


class AgentQLLoader(BaseLoader):
    """
    AgentQLLoader document loader integration

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders.agentql import AgentQLLoader

            loader = AgentQLLoader(
                api_key = "your-api-key"
                url = "url-to-extract-data-from"
                query = "agentql-query-to-extract-data-with"
                params = {
                    "is_scroll_to_bottom_enabled": True
                    # Optional additional parameters
                }
            )
    Lazy load:
        .. code-block:: python

            docs_lazy = loader.lazy_load()
    """

    def __init__(
        self,
        url: str,
        query: str,
        *,
        api_key: str,
        params: Optional[dict] = None,
    ):
        """
        Initialize with API key and params.

        Args:
            url (str): URL to scrape or crawl.
            query (Optional[str]): AgentQL query used to specify the scraped data.
            api_key (str): AgentQL API key. You can create one at https://dev.agentql.com.
            params (Optional[dict]): Additional parameters to pass to the AgentQL API.

            The following parameters are supported:
            wait_for (number): Wait time in seconds for page load (max 10 seconds). Defaults to 0.
            is_scroll_to_bottom_enabled (boolean): Enable/disable scrolling to bottom before snapshot. Defaults to false.
            mode (str): Specifies the extraction mode: standard for complex or high-volume data, or fast for typical use cases. 
            Defaults to fast. You can read more about the mode options in [Guide](https://docs.agentql.com/speed/fast-mode).

            Visit https://docs.agentql.com/rest-api/api-reference for more details.
        """
        self.url = url
        self.query = query
        self.api_key = api_key
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        payload = {"url": self.url, "query": self.query, "params": self.params}

        headers = {"X-API-Key": f"{self.api_key}", "Content-Type": "application/json"}

        try:
            response = httpx.post(
                QUERY_DATA_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            response = e.response
            if response.status_code in [401, 403]:
                raise ValueError(
                    "Please, provide a valid API Key. You can create one at https://dev.agentql.com."
                ) from e
            else:
                try:
                    error_json = response.json()
                    msg = (
                        error_json["error_info"]
                        if "error_info" in error_json
                        else error_json["detail"]
                    )
                except (ValueError, TypeError):
                    msg = f"HTTP {e}."
                raise ValueError(msg) from e
        else:
            data = response.json()
            yield Document(
                page_content=str(data["data"]),
                metadata=data["metadata"],
            )
