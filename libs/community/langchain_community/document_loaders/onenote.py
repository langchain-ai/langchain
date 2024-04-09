"""Loads data from OneNote Notebooks"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
    BaseModel,
    BaseSettings,
    Field,
    FilePath,
    SecretStr,
)

from langchain_community.document_loaders.base import BaseLoader


class _OneNoteGraphSettings(BaseSettings):
    client_id: str = Field(..., env="MS_GRAPH_CLIENT_ID")
    client_secret: SecretStr = Field(..., env="MS_GRAPH_CLIENT_SECRET")

    class Config:
        """Config for OneNoteGraphSettings."""

        env_prefix = ""
        case_sentive = False
        env_file = ".env"


class OneNoteLoader(BaseLoader, BaseModel):
    """Load pages from OneNote notebooks."""

    settings: _OneNoteGraphSettings = Field(default_factory=_OneNoteGraphSettings)
    """Settings for the Microsoft Graph API client."""
    auth_with_token: bool = False
    """Whether to authenticate with a token or not. Defaults to False."""
    access_token: str = ""
    """Personal access token"""
    onenote_api_base_url: str = "https://graph.microsoft.com/v1.0/me/onenote"
    """URL of Microsoft Graph API for OneNote"""
    authority_url = "https://login.microsoftonline.com/consumers/"
    """A URL that identifies a token authority"""
    token_path: FilePath = Path.home() / ".credentials" / "onenote_graph_token.txt"
    """Path to the file where the access token is stored"""
    notebook_name: Optional[str] = None
    """Filter on notebook name"""
    section_name: Optional[str] = None
    """Filter on section name"""
    page_title: Optional[str] = None
    """Filter on section name"""
    object_ids: Optional[List[str]] = None
    """ The IDs of the objects to load data from."""

    def lazy_load(self) -> Iterator[Document]:
        """
        Get pages from OneNote notebooks.

        Returns:
            A list of Documents with attributes:
                - page_content
                - metadata
                    - title
        """
        self._auth()

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install bs4`"
            )

        if self.object_ids is not None:
            for object_id in self.object_ids:
                page_content_html = self._get_page_content(object_id)
                soup = BeautifulSoup(page_content_html, "html.parser")
                page_title = ""
                title_tag = soup.title
                if title_tag:
                    page_title = title_tag.get_text(strip=True)
                page_content = soup.get_text(separator="\n", strip=True)
                yield Document(
                    page_content=page_content, metadata={"title": page_title}
                )
        else:
            request_url = self._url

            while request_url != "":
                response = requests.get(request_url, headers=self._headers, timeout=10)
                response.raise_for_status()
                pages = response.json()

                for page in pages["value"]:
                    page_id = page["id"]
                    page_content_html = self._get_page_content(page_id)
                    soup = BeautifulSoup(page_content_html, "html.parser")
                    page_title = ""
                    title_tag = soup.title
                    if title_tag:
                        page_content = soup.get_text(separator="\n", strip=True)
                    yield Document(
                        page_content=page_content, metadata={"title": page_title}
                    )

                if "@odata.nextLink" in pages:
                    request_url = pages["@odata.nextLink"]
                else:
                    request_url = ""

    def _get_page_content(self, page_id: str) -> str:
        """Get page content from OneNote API"""
        request_url = self.onenote_api_base_url + f"/pages/{page_id}/content"
        response = requests.get(request_url, headers=self._headers, timeout=10)
        response.raise_for_status()
        return response.text

    @property
    def _headers(self) -> Dict[str, str]:
        """Return headers for requests to OneNote API"""
        return {
            "Authorization": f"Bearer {self.access_token}",
        }

    @property
    def _scopes(self) -> List[str]:
        """Return required scopes."""
        return ["Notes.Read"]

    def _auth(self) -> None:
        """Authenticate with Microsoft Graph API"""
        if self.access_token != "":
            return

        if self.auth_with_token:
            with self.token_path.open("r") as token_file:
                self.access_token = token_file.read()
        else:
            try:
                from msal import ConfidentialClientApplication
            except ImportError as e:
                raise ImportError(
                    "MSAL package not found, please install it with `pip install msal`"
                ) from e

            client_instance = ConfidentialClientApplication(
                client_id=self.settings.client_id,
                client_credential=self.settings.client_secret.get_secret_value(),
                authority=self.authority_url,
            )

            authorization_request_url = client_instance.get_authorization_request_url(
                self._scopes
            )
            print("Visit the following url to give consent:")  # noqa: T201
            print(authorization_request_url)  # noqa: T201
            authorization_url = input("Paste the authenticated url here:\n")

            authorization_code = authorization_url.split("code=")[1].split("&")[0]
            access_token_json = client_instance.acquire_token_by_authorization_code(
                code=authorization_code, scopes=self._scopes
            )
            self.access_token = access_token_json["access_token"]

            try:
                if not self.token_path.parent.exists():
                    self.token_path.parent.mkdir(parents=True)
            except Exception as e:
                raise Exception(
                    f"Could not create the folder {self.token_path.parent} "
                    + "to store the access token."
                ) from e

            with self.token_path.open("w") as token_file:
                token_file.write(self.access_token)

    @property
    def _url(self) -> str:
        """Create URL for getting page ids from the OneNoteApi API."""
        query_params_list = []
        filter_list = []
        expand_list = []

        query_params_list.append("$select=id")
        if self.notebook_name is not None:
            filter_list.append(
                "parentNotebook/displayName%20eq%20"
                + f"'{self.notebook_name.replace(' ', '%20')}'"
            )
            expand_list.append("parentNotebook")
        if self.section_name is not None:
            filter_list.append(
                "parentSection/displayName%20eq%20"
                + f"'{self.section_name.replace(' ', '%20')}'"
            )
            expand_list.append("parentSection")
        if self.page_title is not None:
            filter_list.append(
                "title%20eq%20" + f"'{self.page_title.replace(' ', '%20')}'"
            )

        if len(expand_list) > 0:
            query_params_list.append("$expand=" + ",".join(expand_list))
        if len(filter_list) > 0:
            query_params_list.append("$filter=" + "%20and%20".join(filter_list))

        query_params = "&".join(query_params_list)
        if query_params != "":
            query_params = "?" + query_params
        return f"{self.onenote_api_base_url}/pages{query_params}"
