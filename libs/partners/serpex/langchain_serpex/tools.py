"""SERPEX Search Tool for LangChain."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator


class SerpexSearchResults(BaseTool):
    """Tool for searching the web using the SERPEX API.

    SERPEX provides multi-engine search results from Google, Bing, DuckDuckGo,
    Brave, Yahoo, and Yandex search engines in JSON format.

    Setup:
        Install `langchain-serpex` and set environment variable `SERPEX_API_KEY`.

        ```bash
        pip install -U langchain-serpex
        export SERPEX_API_KEY="your-serpex-api-key"
        ```

    Instantiation:
        ```python
        from langchain_serpex import SerpexSearchResults

        # With explicit API key
        tool = SerpexSearchResults(
            api_key="your-serpex-api-key",
            engine="auto",  # or google, bing, duckduckgo, brave, yahoo, yandex
            time_range="day"  # optional: all, day, week, month, year
        )

        # Or using environment variable
        tool = SerpexSearchResults()
        ```

    Invocation:
        ```python
        # Basic search
        results = tool.invoke("latest AI developments")
        print(results)

        # With specific parameters
        results = tool.invoke({
            "query": "Python programming",
            "engine": "google",
            "time_range": "week"
        })
        ```

    Example with Agent:
        ```python
        from langchain_serpex import SerpexSearchResults
        from langchain_openai import ChatOpenAI
        from langchain.agents import initialize_agent, AgentType

        search = SerpexSearchResults(api_key="your-key")
        llm = ChatOpenAI(temperature=0)

        agent = initialize_agent(
            tools=[search],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

        result = agent.run("What's the latest news about AI?")
        ```
    """

    name: str = "serpex_search"
    description: str = (
        "A powerful multi-engine web search tool. "
        "Useful for answering questions about current events, "
        "finding information from the web, and getting real-time data. "
        "Input should be a search query string. "
        "Supports automatic routing with retry logic and multiple search engines "
        "(Google, Bing, DuckDuckGo, Brave, Yahoo, Yandex)."
    )

    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    engine: str = Field(
        default="auto",
        description=(
            "Search engine: auto, google, bing, duckduckgo, brave, yahoo, yandex"
        ),
    )
    category: str = Field(
        default="web",
        description="Search category (currently only 'web' supported)",
    )
    time_range: str | None = Field(
        default=None,
        description=(
            "Time range: all, day, week, month, year (not supported by Brave)"
        ),
    )

    base_url: str = Field(default="https://api.serpex.dev")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that API key exists in environment."""
        api_key = values.get("api_key")
        if not api_key or (
            isinstance(api_key, SecretStr) and not api_key.get_secret_value()
        ):
            api_key_from_env = os.getenv("SERPEX_API_KEY", "")
            if api_key_from_env:
                values["api_key"] = SecretStr(api_key_from_env)
        elif isinstance(api_key, str):
            values["api_key"] = SecretStr(api_key)

        return values

    def _build_params(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Build parameters for the API request."""
        params: dict[str, Any] = {
            "q": query,
            "engine": kwargs.get("engine", self.engine),
            "category": kwargs.get("category", self.category),
        }

        # Add time_range if specified
        time_range = kwargs.get("time_range") or self.time_range
        if time_range is not None:
            params["time_range"] = time_range

        return params

    def _format_results(self, data: dict[str, Any]) -> str:
        """Format the search results into a readable string."""
        results_parts: list[str] = []

        # Instant answers (from knowledge panels/answer boxes)
        if (
            "answers" in data
            and isinstance(data["answers"], list)
            and len(data["answers"]) > 0
        ):
            answer = data["answers"][0]
            if "answer" in answer and answer["answer"]:
                results_parts.append(f"Answer: {answer['answer']}")
            elif "snippet" in answer and answer["snippet"]:
                results_parts.append(f"Featured Snippet: {answer['snippet']}")

        # Infoboxes (knowledge panels)
        if (
            "infoboxes" in data
            and isinstance(data["infoboxes"], list)
            and len(data["infoboxes"]) > 0
        ):
            infobox = data["infoboxes"][0]
            if "description" in infobox and infobox["description"]:
                results_parts.append(f"Knowledge Panel: {infobox['description']}")

        # Organic search results
        if (
            "results" in data
            and isinstance(data["results"], list)
            and len(data["results"]) > 0
        ):
            num_results = data.get("metadata", {}).get(
                "number_of_results", len(data["results"])
            )
            results_parts.append(f"\nFound {num_results} results:\n")

            for i, result in enumerate(data["results"][:10], 1):
                title = result.get("title", "")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                published_date = result.get("published_date")

                result_text = f"[{i}] {title}"
                if url:
                    result_text += f"\nURL: {url}"
                if snippet:
                    result_text += f"\n{snippet}"
                if published_date:
                    result_text += f"\nPublished: {published_date}"

                results_parts.append(result_text)

        # Search suggestions
        if (
            not results_parts
            and "suggestions" in data
            and isinstance(data["suggestions"], list)
            and len(data["suggestions"]) > 0
        ):
            results_parts.append("No direct results found. Related searches:")
            results_parts.extend(data["suggestions"])

        # Query corrections
        if (
            not results_parts
            and "corrections" in data
            and isinstance(data["corrections"], list)
            and len(data["corrections"]) > 0
        ):
            results_parts.append(f"Did you mean: {', '.join(data['corrections'])}?")

        if not results_parts:
            return "No search results found."

        return "\n\n".join(results_parts)

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute the search."""
        params = self._build_params(query, **kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/api/search"

        try:
            with httpx.Client() as client:
                response = client.get(url, params=params, headers=headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    return f"SERPEX API error: {data['error']}"

                return self._format_results(data)

        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Request error occurred: {str(e)}"
        except Exception as e:
            return f"Error searching with SERPEX: {str(e)}"

    async def _arun(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute the search asynchronously."""
        params = self._build_params(query, **kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/api/search"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, params=params, headers=headers, timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    return f"SERPEX API error: {data['error']}"

                return self._format_results(data)

        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Request error occurred: {str(e)}"
        except Exception as e:
            return f"Error searching with SERPEX: {str(e)}"


__all__ = ["SerpexSearchResults"]
