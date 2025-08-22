import json
import os
from typing import Any, Dict, List, Optional

from notion_client import Client
from notion_client.errors import APIResponseError


class NotionWrapper:
    """
    Wrapper around the Notion SDK for basic page operations.
    Reads NOTION_API_KEY / NOTION_DATABASE_ID from env if not passed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_db_id: Optional[str] = None,
        *,
        read_only: bool = False,
        timeout: int = 30,
    ) -> None:
        self.api_key = (api_key or os.getenv("NOTION_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("NOTION_API_KEY is required. Pass api_key=... or set the env var.")

        # Configure client (you can add logging, proxies, etc. here if needed)
        self.client = Client(auth=self.api_key)

        self.default_db_id = (
            default_db_id or os.getenv("NOTION_DATABASE_ID") or ""
        ).strip() or None
        self.read_only = read_only

    # ---------- Internal helpers ----------

    def _title_from_page(self, page_obj: Dict[str, Any]) -> str:
        """Return the concatenated title text from a Notion page object."""
        props = page_obj.get("properties", {}) or {}
        for prop in props.values():
            if prop.get("type") == "title":
                parts = prop.get("title") or []
                # Join all title fragments, not just the first
                return "".join(p.get("plain_text", "") for p in parts if isinstance(p, dict))
        return ""

    def _ensure_parent(self, parent: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure a parent dict exists, defaulting to the default database if configured."""
        if parent:
            return parent
        if self.default_db_id:
            return {"database_id": self.default_db_id}
        raise ValueError(
            "A parent is required (database_id or page_id). Set NOTION_DATABASE_ID or pass parent=..."
        )

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("Write operation blocked: NotionWrapper is in read_only mode.")

    # ---------- Public API ----------

    def search_pages(self, query: str, page_size: int = 5) -> str:
        """
        Search for Notion pages matching the query string.
        Returns a JSON string: {"count": int, "results": [{id,title,url}, ...]}
        """
        page_size = max(1, min(int(page_size), 100))  # Notion caps at 100
        try:
            resp = self.client.search(
                query=query,
                page_size=page_size,
                filter={"value": "page", "property": "object"},
            )
            results = resp.get("results", []) or []
            items: List[Dict[str, Any]] = [
                {"id": r.get("id"), "title": self._title_from_page(r), "url": r.get("url")}
                for r in results
            ]
            return json.dumps({"count": len(items), "results": items})
        except APIResponseError as e:
            return json.dumps({"error": True, "message": str(e), "code": getattr(e, "code", None)})

    def get_page(self, page_id: str) -> str:
        """
        Retrieve a Notion page by its ID.
        Returns a JSON string: {"id", "title", "url", "properties"}
        """
        page_id = (page_id or "").strip()
        if not page_id:
            return json.dumps({"error": True, "message": "page_id is required"})
        try:
            page = self.client.pages.retrieve(page_id=page_id)
            return json.dumps(
                {
                    "id": page.get("id"),
                    "title": self._title_from_page(page),
                    "url": page.get("url"),
                    "properties": page.get("properties", {}) or {},
                }
            )
        except APIResponseError as e:
            return json.dumps({"error": True, "message": str(e), "code": getattr(e, "code", None)})

    def create_page(
        self,
        parent: Optional[Dict[str, Any]],
        properties: Dict[str, Any],
        # children: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Create a new Notion page under the given parent (database or page).
        `parent` must be like {"database_id": "..."} or {"page_id": "..."}.
        `properties` should include a title property when parent is a page.
        Returns: {"id","title","url"}
        """
        self._ensure_writable()
        try:
            parent_obj = self._ensure_parent(parent)
            page = self.client.pages.create(parent=parent_obj, properties=properties)
            return json.dumps(
                {"id": page.get("id"), "title": self._title_from_page(page), "url": page.get("url")}
            )
        except APIResponseError as e:
            return json.dumps({"error": True, "message": str(e), "code": getattr(e, "code", None)})

    def create_page_under_page(
        self,
        parent_page_id: str,
        title: str,
        *,
        children: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Convenience: create a page under a parent page by ID with a simple title.
        """
        self._ensure_writable()
        parent_page_id = (parent_page_id or "").strip()
        if not parent_page_id:
            return json.dumps({"error": True, "message": "parent_page_id is required"})
        try:
            page = self.client.pages.create(
                parent={"page_id": parent_page_id},
                properties={"title": [{"type": "text", "text": {"content": title}}]},
                children=children,
            )
            return json.dumps(
                {"id": page.get("id"), "title": self._title_from_page(page), "url": page.get("url")}
            )
        except APIResponseError as e:
            return json.dumps({"error": True, "message": str(e), "code": getattr(e, "code", None)})

    def update_page(self, page_id: str, properties: Dict[str, Any]) -> str:
        """
        Update properties of an existing Notion page.
        Returns: {"id","title","url"}
        """
        self._ensure_writable()
        page_id = (page_id or "").strip()
        if not page_id:
            return json.dumps({"error": True, "message": "page_id is required"})
        try:
            page = self.client.pages.update(page_id=page_id, properties=properties)
            return json.dumps(
                {"id": page.get("id"), "title": self._title_from_page(page), "url": page.get("url")}
            )
        except APIResponseError as e:
            return json.dumps({"error": True, "message": str(e), "code": getattr(e, "code", None)})
