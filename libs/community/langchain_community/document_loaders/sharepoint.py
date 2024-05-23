"""Loader that loads data from Sharepoint Document Library"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

import requests
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field

from langchain_community.document_loaders.base_o365 import (
    O365BaseLoader,
    _FileType,
)
from langchain_community.document_loaders.parsers.registry import get_parser


class SharePointLoader(O365BaseLoader, BaseLoader):
    """Load  from `SharePoint`."""

    document_library_id: str = Field(...)
    """ The ID of the SharePoint document library to load data from."""
    folder_path: Optional[str] = None
    """ The path to the folder to load data from."""
    object_ids: Optional[List[str]] = None
    """ The IDs of the objects to load data from."""
    folder_id: Optional[str] = None
    """ The ID of the folder to load data from."""
    load_auth: Optional[bool] = False
    """ Whether to load authorization identities."""
    token_path: Path = Path.home() / ".credentials" / "o365_token.txt"
    """ The path to the token to make api calls"""
    file_id: Optional[str] = None
    """ The ID of the file for which we need auth identities"""
    site_id: Optional[str] = None
    """ The ID of the Sharepoint site of the user where the file is present """

    @property
    def _file_types(self) -> Sequence[_FileType]:
        """Return supported file types."""
        return _FileType.DOC, _FileType.DOCX, _FileType.PDF

    @property
    def _scopes(self) -> List[str]:
        """Return required scopes."""
        return ["sharepoint", "basic"]

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily. Use this when working at a large scale."""
        try:
            from O365.drive import Drive, Folder
        except ImportError:
            raise ImportError(
                "O365 package not found, please install it with `pip install o365`"
            )
        drive = self._auth().storage().get_drive(self.document_library_id)
        if not isinstance(drive, Drive):
            raise ValueError(f"There isn't a Drive with id {self.document_library_id}.")
        blob_parser = get_parser("default")
        if self.folder_path:
            target_folder = drive.get_item_by_path(self.folder_path)
            if not isinstance(target_folder, Folder):
                raise ValueError(f"There isn't a folder with path {self.folder_path}.")
            for blob in self._load_from_folder(target_folder):
                for parsed_blob in blob_parser.lazy_parse(blob):
                    auth_identities = self.authorized_identities()
                    parsed_blob.metadata["authorized_identities"] = auth_identities
                    yield parsed_blob
        if self.folder_id:
            target_folder = drive.get_item(self.folder_id)
            if not isinstance(target_folder, Folder):
                raise ValueError(f"There isn't a folder with path {self.folder_path}.")
            for blob in self._load_from_folder(target_folder):
                yield from blob_parser.lazy_parse(blob)
        if self.object_ids:
            for blob in self._load_from_object_ids(drive, self.object_ids):
                yield from blob_parser.lazy_parse(blob)
        if not (self.folder_path or self.folder_id or self.object_ids):
            target_folder = drive.get_root_folder()
            if not isinstance(target_folder, Folder):
                raise ValueError("Unable to fetch root folder")
            for blob in self._load_from_folder(target_folder):
                for blob_part in blob_parser.lazy_parse(blob):
                    blob_part.metadata.update(blob.metadata)
                    yield blob_part

    def authorized_identities(self) -> List:
        data = self._fetch_access_token()
        access_token = data.get("access_token")
        url = (
            f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/"
            f"drives/{self.document_library_id}/items/{self.file_id}/permissions"
        )
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.request("GET", url, headers=headers, data={})
        groups_list = response.json()

        group_names = []

        for group_data in groups_list.get("value"):
            if group_data.get("grantedToV2"):
                if group_data.get("grantedToV2").get("siteGroup"):
                    site_data = group_data.get("grantedToV2").get("siteGroup")
                    # print(group_data)
                    group_names.append(site_data.get("displayName"))
                elif group_data.get("grantedToV2").get("group") or (
                    group_data.get("grantedToV2").get("user")
                ):
                    site_data = group_data.get("grantedToV2").get("group") or (
                        group_data.get("grantedToV2").get("user")
                    )
                    # print(group_data)
                    group_names.append(site_data.get("displayName"))

        return group_names

    def _fetch_access_token(self) -> Any:
        with open(self.token_path) as f:
            s = f.read()
        data = json.loads(s)
        return data
