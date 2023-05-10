"""NotionAPI to save a document.

Future plans:
- Improve the saving process, title, fields, properties
- Add support for reading the database
"""
from typing import Dict, Optional, Any

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class NotionAPIWrapper(BaseModel):
    """Wrapper around Notion API.

    TODO: Improve docs

    To use, you should have the ``notion-client` python package installed,
    and the environment variable ``NOTION_TOKEN`` AND ``NOTION_DATABASE_ID`` set, or pass
    `notion_token` as a named parameter to the constructor.

    To get your token:

    1- Go to Notion settings, create a new integration and save the integration secret
     to NOTION_API_TOKEN
    2- Create a new database in notion and copy its ID to NOTION_DATABASE_ID.
    The id is normally the last part of the database URL. See
    https://developers.notion.com/reference/retrieve-a-database

    """

    notion_client: Any  #: :meta private:
    """Notion API client."""
    notion_api_token: Optional[str] = None
    """Notion API token. Can be passed in directly or stored as environment variable
    NOTION_API_TOKEN. To get your token: go to Notion settings and create a new 
    integration. See 
    https://developers.notion.com/docs/authorization#internal-integration-auth-flow-set-up
    """
    notion_database_id: Optional[str] = None
    """ID for a specific database in your Notion workspace. The ID is normally the last
    part of the database URL. See 
    https://developers.notion.com/docs/authorization#internal-integration-auth-flow-set-up
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        notion_token = get_from_dict_or_env(values, "notion_api_token", "NOTION_API_TOKEN")
        notion_database_id = get_from_dict_or_env(
            values, "notion_database_id", "NOTION_DATABASE_ID"
        )
        values["notion_database_id"] = notion_database_id

        try:
            from notion_client import Client
        except ImportError:
            raise ValueError(
                "Could not import notion_client python package. "
                "Please it install it with `pip install notion-client`"
            )
        values["notion_client"] = Client({"auth": notion_token})
        return values

    @property
    def _parent(self) -> Dict[str, str]:
        return {"database_id": self.notion_database_id}

    def _get_properties(self, title: str) -> Dict[str, Any]:
        return {"Name": {"title": [{"type": "text", "text": {"content": title}}]}}

    def _get_paragraph(self, content: str) -> Dict[str, Any]:
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]}
        }

    def run(self, document: str) -> None:
        """Saves document to Notion."""
        # TODO: change how the title is set
        title = document[:10]
        properties = self._get_properties(title)
        paragraph = self._get_paragraph(document)
        self.notion_client.pages.create(
            parent=self._parent, properties=properties, children=[paragraph]
        )
