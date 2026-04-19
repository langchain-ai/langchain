"""AgentDB retriever — real-time curated knowledge for AI agents."""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

_BASE_URL = "https://agentdb-production-9ba0.up.railway.app"
_LATEST_PATH = "/v1/knowledge/latest"
_SEARCH_PATH = "/v1/knowledge/search"


class AgentDBRetriever(BaseRetriever):
    """AgentDB retriever for real-time, pre-summarised knowledge.

    AgentDB is a curated knowledge API for AI agents. It continuously ingests
    YouTube channels, podcasts, and blog feeds on a Monday/Wednesday/Friday
    schedule, summarises each item with Claude, and exposes the results via a
    structured REST API with optional semantic vector search.

    Setup:
        Sign up for a free API key at https://agentdb.dev, then set the
        ``AGENTDB_API_KEY`` environment variable or pass ``api_key``::

            .. code-block:: bash

                export AGENTDB_API_KEY="agentdb-..."

        Install the package::

            .. code-block:: bash

                pip install -U langchain-agentdb

    Instantiate:
        .. code-block:: python

            from langchain_agentdb import AgentDBRetriever

            # Semantic search (requires Pro tier)
            retriever = AgentDBRetriever(mode="search", k=10)

            # Latest items with optional filters (free tier)
            retriever = AgentDBRetriever(
                mode="latest",
                k=20,
                content_type="podcast",
                tags="ai,machine-learning",
                min_confidence=0.7,
            )

    Usage:
        .. code-block:: python

            docs = retriever.invoke("AI safety developments this week")
            for doc in docs:
                print(doc.metadata["title"])
                print(doc.page_content[:200])

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            def format_docs(docs):
                return "\\n\\n".join(
                    f"**{d.metadata['title']}**\\n{d.page_content}" for d in docs
                )

            llm = ChatOpenAI(model="gpt-4o-mini")
            retriever = AgentDBRetriever()

            prompt = ChatPromptTemplate.from_template(
                "Answer using this recent knowledge:\\n{context}\\n\\nQuestion: {question}"
            )
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            answer = chain.invoke("What are the latest AI developments?")

    Each returned ``Document`` contains:

    - ``page_content``: 2-3 paragraph summary plus key bullet points
    - ``metadata["title"]``: clean, descriptive title
    - ``metadata["content_type"]``: ``article``, ``video``, ``podcast``, ``data``,
      or ``research``
    - ``metadata["tags"]``: list of topic tags (e.g. ``["ai-safety", "deepmind"]``)
    - ``metadata["confidence"]``: Claude's confidence score (0.0–1.0)
    - ``metadata["published_at"]``: ISO timestamp string
    - ``metadata["source_url"]``: original source URL
    - ``metadata["key_points"]``: list of 5-10 concrete takeaways
    """

    api_key: Optional[str] = None
    """AgentDB API key. Falls back to the ``AGENTDB_API_KEY`` environment variable."""

    mode: Literal["search", "latest"] = "search"
    """Retrieval mode.

    - ``"search"``: semantic vector search — uses the query to find the most
      relevant items. Requires a Pro tier key.
    - ``"latest"``: returns the newest items, optionally filtered by
      ``content_type`` or ``tags``. The query string is ignored.
    """

    k: int = 10
    """Maximum number of documents to return (1–50 for search, 1–100 for latest)."""

    content_type: Optional[Literal["article", "video", "podcast", "data", "research"]] = None
    """Filter by content type. Only applied in ``mode="latest"``."""

    tags: Optional[str] = None
    """Comma-separated tag filter, e.g. ``"ai,machine-learning"``.
    Uses array-overlap semantics — any matching tag is included.
    Only applied in ``mode="latest"``."""

    min_confidence: Optional[float] = None
    """Minimum confidence score (0.0–1.0). Items below this threshold are
    excluded. Confidence is assigned by Claude at ingestion time."""

    base_url: str = _BASE_URL
    """AgentDB API base URL. Override for self-hosted deployments."""

    timeout: float = 10.0
    """HTTP request timeout in seconds."""

    # ── internal helpers ────────────────────────────────────────────────────

    def _resolved_api_key(self) -> str:
        key = self.api_key or os.environ.get("AGENTDB_API_KEY", "")
        if not key:
            msg = (
                "No AgentDB API key found. Pass api_key= or set the "
                "AGENTDB_API_KEY environment variable. "
                "Get a free key at https://agentdb.dev"
            )
            raise ValueError(msg)
        return key

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._resolved_api_key()}",
            "Accept": "application/json",
        }

    def _build_params(self, query: str) -> tuple[str, dict[str, Any]]:
        """Return (url, params) for the current mode."""
        if self.mode == "search":
            if not query.strip():
                msg = (
                    "A non-empty query string is required for mode='search'. "
                    "Use mode='latest' to fetch recent items without a query."
                )
                raise ValueError(msg)
            url = f"{self.base_url}{_SEARCH_PATH}"
            params: dict[str, Any] = {"q": query, "limit": min(self.k, 50)}
        else:
            url = f"{self.base_url}{_LATEST_PATH}"
            params = {"limit": min(self.k, 100), "page": 1}
            if self.content_type:
                params["content_type"] = self.content_type
            if self.tags:
                params["tags"] = self.tags
        return url, params

    def _items_to_documents(self, items: list[dict[str, Any]]) -> list[Document]:
        docs = []
        for item in items:
            confidence = float(item.get("confidence") or 0.0)
            if self.min_confidence is not None and confidence < self.min_confidence:
                continue

            body = item.get("body") or {}
            key_points: list[str] = body.get("key_points", [])

            content_parts = [item.get("summary", "").strip()]
            if key_points:
                content_parts.append(
                    "Key points:\n" + "\n".join(f"• {p}" for p in key_points)
                )
            page_content = "\n\n".join(p for p in content_parts if p)

            metadata: dict[str, Any] = {
                "id": item.get("id"),
                "title": item.get("title"),
                "content_type": item.get("content_type"),
                "tags": item.get("tags") or [],
                "confidence": confidence,
                "published_at": item.get("published_at"),
                "source_url": body.get("source_url") or item.get("source_url"),
                "key_points": key_points,
            }
            if "similarity" in item:
                metadata["similarity"] = item["similarity"]

            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    # ── sync ────────────────────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        url, params = self._build_params(query)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self._build_headers(), params=params)
        response.raise_for_status()
        return self._items_to_documents(response.json().get("items", []))

    # ── async ───────────────────────────────────────────────────────────────

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        url, params = self._build_params(query)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                url, headers=self._build_headers(), params=params
            )
        response.raise_for_status()
        return self._items_to_documents(response.json().get("items", []))
