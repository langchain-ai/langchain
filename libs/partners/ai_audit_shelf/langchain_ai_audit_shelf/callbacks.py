"""Callback handler for AI Audit Shelf."""

from typing import Any
from uuid import UUID

import httpx
import requests
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AIAuditCallbackHandler(BaseCallbackHandler):
    """Logs every LLM and tool call to the AI Audit server.

    This callback handler interacts with the AI Audit Shelf REST API, sending
    prompts and results to be recorded as immutable chapters.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        actor: str = "langchain-agent",
        source: str = "langchain",
        api_key: str | None = None,
    ) -> None:
        """Initialize the AI Audit Callback Handler.

        Args:
            api_url: The base URL of the AI Audit server.
            actor: The actor name to log in the audit trail.
            source: The source system identifier.
            api_key: Optional API key if the audit server requires it.
        """
        self.api_url = api_url.rstrip("/")
        self.actor = actor
        self.source = source
        self.api_key = api_key
        self._chapters: list[str] = []

    def _get_headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _log_chapter(self, prompt: str, result: str) -> str | None:
        """Log a synchronous chapter to the audit server."""
        try:
            params = {
                "prompt": prompt,
                "result": result,
                "actor": self.actor,
                "source": self.source,
            }
            response = requests.post(
                f"{self.api_url}/chapter",
                params=params,
                headers=self._get_headers(),
                timeout=10.0,
            )
            response.raise_for_status()
        except Exception:  # noqa: BLE001
            # We don't want to break the main application if the audit logging fails
            return None
        else:
            chapter_id: str = response.json()["chapter"]["id"]
            self._chapters.append(chapter_id)
            return chapter_id

    async def _log_chapter_async(self, prompt: str, result: str) -> str | None:
        """Log an asynchronous chapter to the audit server."""
        try:
            params = {
                "prompt": prompt,
                "result": result,
                "actor": self.actor,
                "source": self.source,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/chapter",
                    params=params,
                    headers=self._get_headers(),
                    timeout=10.0,
                )
                response.raise_for_status()
        except Exception:  # noqa: BLE001
            return None
        else:
            chapter_id: str = response.json()["chapter"]["id"]
            self._chapters.append(chapter_id)
            return chapter_id

    def bundle(self, title: str, feature: str = "") -> dict[str, Any] | None:
        """Bundle all logged chapters into a book.

        Args:
            title: The title of the book.
            feature: The feature category (defaults to title).

        Returns:
            The created book representation, or None if the request fails.
        """
        try:
            params = {
                "title": title,
                "feature": feature or title,
            }
            response = requests.post(
                f"{self.api_url}/book",
                params=params,
                json=self._chapters,
                headers=self._get_headers(),
                timeout=10.0,
            )
            response.raise_for_status()
        except Exception:  # noqa: BLE001
            return None
        else:
            book_data: dict[str, Any] = response.json()["book"]
            return book_data

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Run when LLM finishes executing."""
        if not response.generations:
            return

        # We try to extract the prompt from kwargs if available
        prompt = str(kwargs.get("prompts", [""]))
        if isinstance(kwargs.get("prompts"), list) and kwargs["prompts"]:
            prompt = kwargs["prompts"][0]

        result = response.generations[0][0].text
        self._log_chapter(prompt=prompt, result=result)

    async def on_llm_end_async(
        self,
        response: LLMResult,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Run when LLM finishes executing asynchronously."""
        if not response.generations:
            return

        prompt = str(kwargs.get("prompts", [""]))
        if isinstance(kwargs.get("prompts"), list) and kwargs["prompts"]:
            prompt = kwargs["prompts"][0]

        result = response.generations[0][0].text
        await self._log_chapter_async(prompt=prompt, result=result)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        tool_name = kwargs.get("name", "unknown")
        result = str(output)[:2000]
        self._log_chapter(prompt=f"Tool: {tool_name}", result=result)

    async def on_tool_end_async(
        self,
        output: Any,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running asynchronously."""
        tool_name = kwargs.get("name", "unknown")
        result = str(output)[:2000]
        await self._log_chapter_async(prompt=f"Tool: {tool_name}", result=result)
