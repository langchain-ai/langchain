# Prerequisites:
# 1. Create a Dropbox app.
# 2. Give the app these scope permissions: `files.metadata.read`
#    and `files.content.read`.
# 3. Generate access token: https://www.dropbox.com/developers/apps/create.
# 4. `pip install dropbox` (requires `pip install unstructured[pdf]` for PDF filetype).


import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, model_validator

from langchain_community.document_loaders.base import BaseLoader


class DropboxLoader(BaseLoader, BaseModel):
    """Load files from `Dropbox`.

    In addition to common files such as text and PDF files, it also supports
    *Dropbox Paper* files.
    """

    dropbox_access_token: str
    """Dropbox access token."""
    dropbox_folder_path: Optional[str] = None
    """The folder path to load from."""
    dropbox_file_paths: Optional[List[str]] = None
    """The file paths to load from."""
    recursive: bool = False
    """Flag to indicate whether to load files recursively from subfolders."""

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, values: Dict[str, Any]) -> Any:
        """Validate that either folder_path or file_paths is set, but not both."""
        if (
            values.get("dropbox_folder_path") is not None
            and values.get("dropbox_file_paths") is not None
        ):
            raise ValueError("Cannot specify both folder_path and file_paths")
        if values.get("dropbox_folder_path") is None and not values.get(
            "dropbox_file_paths"
        ):
            raise ValueError("Must specify either folder_path or file_paths")

        return values

    def _create_dropbox_client(self) -> Any:
        """Create a Dropbox client."""
        try:
            from dropbox import Dropbox, exceptions
        except ImportError:
            raise ImportError("You must run `pip install dropbox")

        try:
            dbx = Dropbox(self.dropbox_access_token)
            dbx.users_get_current_account()
        except exceptions.AuthError as ex:
            raise ValueError(
                "Invalid Dropbox access token. Please verify your token and try again."
            ) from ex
        return dbx

    def _load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """Load documents from a Dropbox folder."""
        dbx = self._create_dropbox_client()

        try:
            from dropbox import exceptions
            from dropbox.files import FileMetadata
        except ImportError:
            raise ImportError("You must run `pip install dropbox")

        try:
            results = dbx.files_list_folder(folder_path, recursive=self.recursive)
        except exceptions.ApiError as ex:
            raise ValueError(
                f"Could not list files in the folder: {folder_path}. "
                "Please verify the folder path and try again."
            ) from ex

        files = [entry for entry in results.entries if isinstance(entry, FileMetadata)]
        documents = [
            doc
            for doc in (self._load_file_from_path(file.path_display) for file in files)
            if doc is not None
        ]
        return documents

    def _load_file_from_path(self, file_path: str) -> Optional[Document]:
        """Load a file from a Dropbox path."""
        dbx = self._create_dropbox_client()

        try:
            from dropbox import exceptions
        except ImportError:
            raise ImportError("You must run `pip install dropbox")

        try:
            file_metadata = dbx.files_get_metadata(file_path)

            if file_metadata.is_downloadable:
                _, response = dbx.files_download(file_path)

            # Some types such as Paper, need to be exported.
            elif file_metadata.export_info:
                _, response = dbx.files_export(file_path, "markdown")

        except exceptions.ApiError as ex:
            raise ValueError(
                f"Could not load file: {file_path}. Please verify the file path"
                "and try again."
            ) from ex

        try:
            text = response.content.decode("utf-8")
        except UnicodeDecodeError:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".pdf":
                print(f"File {file_path} type detected as .pdf")  # noqa: T201
                from langchain_community.document_loaders import UnstructuredPDFLoader

                # Download it to a temporary file.
                temp_dir = tempfile.TemporaryDirectory()
                temp_pdf = Path(temp_dir.name) / "tmp.pdf"
                with open(temp_pdf, mode="wb") as f:
                    f.write(response.content)

                try:
                    loader = UnstructuredPDFLoader(str(temp_pdf))
                    docs = loader.load()
                    if docs:
                        return docs[0]
                except Exception as pdf_ex:
                    print(f"Error while trying to parse PDF {file_path}: {pdf_ex}")  # noqa: T201
                    return None
            else:
                print(  # noqa: T201
                    f"File {file_path} could not be decoded as pdf or text. Skipping."
                )

            return None

        metadata = {
            "source": f"dropbox://{file_path}",
            "title": os.path.basename(file_path),
        }
        return Document(page_content=text, metadata=metadata)

    def _load_documents_from_paths(self) -> List[Document]:
        """Load documents from a list of Dropbox file paths."""
        if not self.dropbox_file_paths:
            raise ValueError("file_paths must be set")

        return [
            doc
            for doc in (
                self._load_file_from_path(file_path)
                for file_path in self.dropbox_file_paths
            )
            if doc is not None
        ]

    def load(self) -> List[Document]:
        """Load documents."""
        if self.dropbox_folder_path is not None:
            return self._load_documents_from_folder(self.dropbox_folder_path)
        else:
            return self._load_documents_from_paths()
