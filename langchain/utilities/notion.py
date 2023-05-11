"""NotionAPI to write a document."""
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.schema import Document
from langchain.utils import get_from_dict_or_env


class NotionAPIWrapper(BaseModel):
    """Wrapper around Notion API.

    To use, you should have the ``notion-client` python package installed,
    and the environment variable ``NOTION_API_TOKEN`` AND ``NOTION_DATABASE_ID`` set,
    or pass `notion_api_token` as a named parameter to the constructor.

    To get your token:
    1. Go to Notion settings, create a new integration and save the integration secret
        to NOTION_API_TOKEN.
    2. Add new integration to a specific Notion database and copy the database ID
        to NOTION_DATABASE_ID. The ID is normally the last part of the database URL. See
        https://developers.notion.com/reference/retrieve-a-database
    """

    notion_api_token: str
    """Notion API token. Can be passed in directly or stored as environment variable
        NOTION_API_TOKEN. To get your token: go to Notion settings and create a new 
        integration. See 
        https://developers.notion.com/docs/authorization#internal-integration-auth-flow-set-up
    """  # noqa: E501
    notion_database_id: str
    """ID for a specific database in your Notion workspace. The ID is normally the last
        part of the database URL. See 
        https://developers.notion.com/docs/authorization#internal-integration-auth-flow-set-up
    """  # noqa: E501
    notion_version: str = "2022-06-28"
    """Notion API version."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["notion_api_token"] = get_from_dict_or_env(
            values, "notion_api_token", "NOTION_API_TOKEN"
        )

        return values

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.notion_api_token}",
            "accept": "application/json",
            "Notion-Version": self.notion_version,
            "content-type": "application/json",
        }

    @property
    def _parent(self) -> Dict[str, str]:
        return {"database_id": self.notion_database_id}

    # TODO: Support more property types.
    def _get_properties(
        self, title: str, **text_properties: Dict[str, str]
    ) -> Dict[str, Any]:
        properties = {
            name: {"rich_text": [{"type": "text", "text": {"content": value}}]}
            for name, value in text_properties.items()
        }
        properties["Name"] = {"title": [{"type": "text", "text": {"content": title}}]}
        return properties

    def _get_paragraph(self, content: str) -> Dict[str, Any]:
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": content}}]
            },
        }

    def _request_and_check(
        self, method: str, url: str, data: Optional[dict] = None
    ) -> dict:
        data = data or {}
        response = requests.request(method, url, headers=self._headers, json=data)
        if response.status_code != 200:
            raise ValueError(
                f"Notion  request failed with status "
                f"{response.status_code}. Full API response:\n{response.text}"
            )
        return response.json()

    def write(
        self,
        text: str,
        title: Optional[str] = None,
        **text_properties: Dict[str, str],
    ) -> None:
        """Write text to a new document in database."""
        title = title or text[:20]
        data = {
            "parent": self._parent,
            "properties": self._get_properties(title, **text_properties),
            "children": [self._get_paragraph(para) for para in text.split("\n")],
        }
        url = "https://api.notion.com/v1/pages"
        self._request_and_check("post", url, data=data)

    # TODO: Support doc metadata.
    def read_page(self, page_id: str) -> Document:
        """Return a page as a document."""
        base_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        start_cursor = ""
        has_more = True
        data = {}
        paragraphs = []
        while has_more:
            url = base_url + start_cursor
            response = self._request_and_check("get", url, data=data)
            paragraphs.extend(
                [
                    b["paragraph"]["rich_text"]
                    for b in response["results"]
                    if "paragraph" in b
                ]
            )
            has_more = response["has_more"]
            start_cursor = f"?start_cursor={response['next_cursor']}"
        content = "\n".join(
            ["".join([rt["plain_text"] for rt in para]) for para in paragraphs]
        )
        doc = Document(page_content=content)
        return doc

    # TODO: Support filtering and sorting.
    def read(self) -> List[Document]:
        """Return all pages in a database as documents."""
        docs: List[Document] = []
        url = f"https://api.notion.com/v1/databases/{self.notion_database_id}/query"
        data = {}
        has_more = True
        while has_more:
            response = self._request_and_check("post", url)
            for res in response["results"]:
                page_id = res["id"]
                docs.append(self.read_page(page_id))
            has_more = response["has_more"]
            data["start_cursor"] = response["next_cursor"]
        return docs
