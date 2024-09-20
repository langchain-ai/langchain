"""Loader that loads data from Sharepoint Document Library"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

import requests  # type: ignore
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pydantic import Field

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
    load_extended_metadata: Optional[bool] = False
    """ Whether to load extended metadata. Size, Owner and full_path."""

    @property
    def _file_types(self) -> Sequence[_FileType]:
        """Return supported file types.
        Returns:
            A sequence of supported file types.
        """
        return _FileType.DOC, _FileType.DOCX, _FileType.PDF

    @property
    def _scopes(self) -> List[str]:
        """Return required scopes.
        Returns:
            List[str]: A list of required scopes.
        """
        return ["sharepoint", "basic"]

    def lazy_load(self) -> Iterator[Document]:
        """
        Load documents lazily. Use this when working at a large scale.
        Yields:
            Document: A document object representing the parsed blob.
        """
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
                file_id = str(blob.metadata.get("id"))
                if self.load_auth is True:
                    auth_identities = self.authorized_identities(file_id)
                if self.load_extended_metadata is True:
                    extended_metadata = self.get_extended_metadata(file_id)
                    extended_metadata.update({"source_full_url": target_folder.web_url})
                for parsed_blob in blob_parser.lazy_parse(blob):
                    if self.load_auth is True:
                        parsed_blob.metadata["authorized_identities"] = auth_identities
                    if self.load_extended_metadata is True:
                        parsed_blob.metadata.update(extended_metadata)
                    yield parsed_blob
        if self.folder_id:
            target_folder = drive.get_item(self.folder_id)
            if not isinstance(target_folder, Folder):
                raise ValueError(f"There isn't a folder with path {self.folder_path}.")
            for blob in self._load_from_folder(target_folder):
                file_id = str(blob.metadata.get("id"))
                if self.load_auth is True:
                    auth_identities = self.authorized_identities(file_id)
                if self.load_extended_metadata is True:
                    extended_metadata = self.get_extended_metadata(file_id)
                    extended_metadata.update({"source_full_url": target_folder.web_url})
                for parsed_blob in blob_parser.lazy_parse(blob):
                    if self.load_auth is True:
                        parsed_blob.metadata["authorized_identities"] = auth_identities
                    if self.load_extended_metadata is True:
                        parsed_blob.metadata.update(extended_metadata)
                    yield parsed_blob
        if self.object_ids:
            for blob in self._load_from_object_ids(drive, self.object_ids):
                file_id = str(blob.metadata.get("id"))
                if self.load_auth is True:
                    auth_identities = self.authorized_identities(file_id)
                if self.load_extended_metadata is True:
                    extended_metadata = self.get_extended_metadata(file_id)
                for parsed_blob in blob_parser.lazy_parse(blob):
                    if self.load_auth is True:
                        parsed_blob.metadata["authorized_identities"] = auth_identities
                    if self.load_extended_metadata is True:
                        parsed_blob.metadata.update(extended_metadata)
                    yield parsed_blob

        if not (self.folder_path or self.folder_id or self.object_ids):
            target_folder = drive.get_root_folder()
            if not isinstance(target_folder, Folder):
                raise ValueError("Unable to fetch root folder")
            for blob in self._load_from_folder(target_folder):
                file_id = str(blob.metadata.get("id"))
                if self.load_auth is True:
                    auth_identities = self.authorized_identities(file_id)
                if self.load_extended_metadata is True:
                    extended_metadata = self.get_extended_metadata(file_id)
                for blob_part in blob_parser.lazy_parse(blob):
                    blob_part.metadata.update(blob.metadata)
                    if self.load_auth is True:
                        blob_part.metadata["authorized_identities"] = auth_identities
                    if self.load_extended_metadata is True:
                        blob_part.metadata.update(extended_metadata)
                        blob_part.metadata.update(
                            {"source_full_url": target_folder.web_url}
                        )
                    yield blob_part

    def authorized_identities(self, file_id: str) -> List:
        """
        Retrieve the access identities (user/group emails) for a given file.
        Args:
            file_id (str): The ID of the file.
        Returns:
            List: A list of group names (email addresses) that have
                  access to the file.
        """
        data = self._fetch_access_token()
        access_token = data.get("access_token")
        url = (
            "https://graph.microsoft.com/v1.0/drives"
            f"/{self.document_library_id}/items/{file_id}/permissions"
        )
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.request("GET", url, headers=headers)
        access_list = response.json()

        group_names = []

        for access_data in access_list.get("value"):
            if access_data.get("grantedToV2"):
                site_data = (
                    (access_data.get("grantedToV2").get("siteUser"))
                    or (access_data.get("grantedToV2").get("user"))
                    or (access_data.get("grantedToV2").get("group"))
                )
                if site_data:
                    email = site_data.get("email")
                    if email:
                        group_names.append(email)
        return group_names

    def _fetch_access_token(self) -> Any:
        """
        Fetch the access token from the token file.
        Returns:
            The access token as a dictionary.
        """
        with open(self.token_path, encoding="utf-8") as f:
            s = f.read()
        data = json.loads(s)
        return data

    def get_extended_metadata(self, file_id: str) -> dict:
        """
        Retrieve extended metadata for a file in SharePoint.
        As of today, following fields are supported in the extended metadata:
        - size: size of the source file.
        - owner: display name of the owner of the source file.
        - full_path: pretty human readable path of the source file.
        Args:
            file_id (str): The ID of the file.
        Returns:
            dict: A dictionary containing the extended metadata of the file,
                  including size, owner, and full path.
        """
        data = self._fetch_access_token()
        access_token = data.get("access_token")
        url = (
            "https://graph.microsoft.com/v1.0/drives/"
            f"{self.document_library_id}/items/{file_id}"
            "?$select=size,createdBy,parentReference,name"
        )
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.request("GET", url, headers=headers)
        metadata = response.json()
        staged_metadata = {
            "size": metadata.get("size", 0),
            "owner": metadata.get("createdBy", {})
            .get("user", {})
            .get("displayName", ""),
            "full_path": metadata.get("parentReference", {})
            .get("path", "")
            .split(":")[-1]
            + "/"
            + metadata.get("name", ""),
        }
        return staged_metadata
