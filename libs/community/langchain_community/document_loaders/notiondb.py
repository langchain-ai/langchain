import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

NOTION_BASE_URL = "https://api.notion.com/v1"
DATABASE_URL = NOTION_BASE_URL + "/databases/{database_id}/query"
PAGE_URL = NOTION_BASE_URL + "/pages/{page_id}"
BLOCK_URL = NOTION_BASE_URL + "/blocks/{block_id}/children"

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class NotionDBLoader(BaseLoader):
    """Load from `Notion DB`.

    Reads content from pages within a Notion Database.
    Args:
        integration_token (str): Notion integration token.
        database_id (str): Notion database id.
        request_timeout_sec (int): Timeout for Notion requests in seconds.
            Defaults to 10.
        filter_object (Dict[str, Any]): Filter object used to limit returned
            entries based on specified criteria.
            E.g.: {
                "timestamp": "last_edited_time",
                "last_edited_time": {
                    "on_or_after": "2024-02-07"
                }
            } -> will only return entries that were last edited
                on or after 2024-02-07
            Notion docs: https://developers.notion.com/reference/post-database-query-filter
            Defaults to None, which will return ALL entries.
    """

    def __init__(
        self,
        integration_token: str,
        database_id: str,
        request_timeout_sec: Optional[int] = 10,
        *,
        filter_object: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize with parameters."""
        if not integration_token:
            raise ValueError("integration_token must be provided")
        if not database_id:
            raise ValueError("database_id must be provided")

        self.token = integration_token
        self.database_id = database_id
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }
        self.request_timeout_sec = request_timeout_sec
        self.filter_object = filter_object or {}

    def load(self) -> List[Document]:
        """Load documents from the Notion database.
        Returns:
            List[Document]: List of documents.
        """
        page_summaries = self._retrieve_page_summaries()
        return list(self.load_page(page_summary) for page_summary in page_summaries)

    def _retrieve_page_summaries(
        self, query_dict: Dict[str, Any] = {"page_size": 100}
    ) -> List[Dict[str, Any]]:
        """
        Get all the pages from a Notion database
        OR filter based on specified criteria.
        """
        pages: List[Dict[str, Any]] = []

        while True:
            data = self._request(
                DATABASE_URL.format(database_id=self.database_id),
                method="POST",
                query_dict=query_dict,
                filter_object=self.filter_object,
            )

            pages.extend(data.get("results"))

            if not data.get("has_more"):
                break

            query_dict["start_cursor"] = data.get("next_cursor")

        return pages

    def load_page(self, page_summary: Dict[str, Any]) -> Document:
        """Read a page.

        Args:
            page_summary: Page summary from Notion API.
        """
        page_id = page_summary["id"]

        # load properties as metadata
        metadata: Dict[str, Any] = {}

        value: Any

        for prop_name, prop_data in page_summary["properties"].items():
            prop_type = prop_data["type"]

            if prop_type == "rich_text":
                value = self._concatenate_rich_text(prop_data["rich_text"])
            elif prop_type == "title":
                value = self._concatenate_rich_text(prop_data["title"])
            elif prop_type == "multi_select":
                value = (
                    [item["name"] for item in prop_data["multi_select"]]
                    if prop_data["multi_select"]
                    else []
                )
            elif prop_type == "url":
                value = prop_data["url"]
            elif prop_type == "unique_id":
                value = (
                    f'{prop_data["unique_id"]["prefix"]}-{prop_data["unique_id"]["number"]}'
                    if prop_data["unique_id"]
                    else None
                )
            elif prop_type == "status":
                value = prop_data["status"]["name"] if prop_data["status"] else None
            elif prop_type == "people":
                value = []
                if prop_data["people"]:
                    for item in prop_data["people"]:
                        name = item.get("name")
                        if not name:
                            logger.warning(
                                "Missing 'name' in 'people' property "
                                f"for page {page_id}"
                            )
                        value.append(name)
            elif prop_type == "date":
                value = prop_data["date"] if prop_data["date"] else None
            elif prop_type == "last_edited_time":
                value = (
                    prop_data["last_edited_time"]
                    if prop_data["last_edited_time"]
                    else None
                )
            elif prop_type == "created_time":
                value = prop_data["created_time"] if prop_data["created_time"] else None
            elif prop_type == "checkbox":
                value = prop_data["checkbox"]
            elif prop_type == "email":
                value = prop_data["email"]
            elif prop_type == "number":
                value = prop_data["number"]
            elif prop_type == "select":
                value = prop_data["select"]["name"] if prop_data["select"] else None
            else:
                value = None

            metadata[prop_name.lower()] = value

        metadata["id"] = page_id

        return Document(page_content=self._load_blocks(page_id), metadata=metadata)

    def _load_blocks(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block and its children."""
        result_lines_arr: List[str] = []
        cur_block_id: str = block_id

        while cur_block_id:
            data = self._request(BLOCK_URL.format(block_id=cur_block_id))

            for result in data["results"]:
                result_obj = result[result["type"]]

                if "rich_text" not in result_obj:
                    continue

                cur_result_text_arr: List[str] = []

                for rich_text in result_obj["rich_text"]:
                    if "text" in rich_text:
                        cur_result_text_arr.append(
                            "\t" * num_tabs + rich_text["text"]["content"]
                        )

                if result["has_children"]:
                    children_text = self._load_blocks(
                        result["id"], num_tabs=num_tabs + 1
                    )
                    cur_result_text_arr.append(children_text)

                result_lines_arr.append("\n".join(cur_result_text_arr))

            cur_block_id = data.get("next_cursor")

        return "\n".join(result_lines_arr)

    def _request(
        self,
        url: str,
        method: str = "GET",
        query_dict: Dict[str, Any] = {},
        *,
        filter_object: Optional[Dict[str, Any]] = None,
    ) -> Any:
        json_payload = query_dict.copy()
        if filter_object:
            json_payload["filter"] = filter_object
        res = requests.request(
            method,
            url,
            headers=self.headers,
            json=json_payload,
            timeout=self.request_timeout_sec,
        )
        res.raise_for_status()
        return res.json()

    def _concatenate_rich_text(self, rich_text_array: List[Dict[str, Any]]) -> str:
        """Concatenate all text content from a rich_text array."""
        return "".join(item["plain_text"] for item in rich_text_array)
