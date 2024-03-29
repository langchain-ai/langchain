# Prerequisites:
# 1. Create a Google Cloud project
# 2. Enable the Google Drive API:
#   https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com
# 3. Authorize credentials for desktop app:
#   https://developers.google.com/drive/api/quickstart/python#authorize_credentials_for_a_desktop_application # noqa: E501
# 4. For service accounts visit
#   https://cloud.google.com/iam/docs/service-accounts-create

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator

from langchain_community.document_loaders.base import BaseLoader

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveLoader(BaseLoader, BaseModel):
    """Load Google Docs from `Google Drive`."""

    service_account_key: Path = Path.home() / ".credentials" / "keys.json"
    """Path to the service account key file."""
    credentials_path: Path = Path.home() / ".credentials" / "credentials.json"
    """Path to the credentials file."""
    token_path: Path = Path.home() / ".credentials" / "token.json"
    """Path to the token file."""
    folder_id: Optional[str] = None
    """The folder id to load from."""
    document_ids: Optional[List[str]] = None
    """The document ids to load from."""
    file_ids: Optional[List[str]] = None
    """The file ids to load from."""
    recursive: bool = False
    """Whether to load recursively. Only applies when folder_id is given."""
    file_types: Optional[Sequence[str]] = None
    """The file types to load. Only applies when folder_id is given."""
    load_trashed_files: bool = False
    """Whether to load trashed files. Only applies when folder_id is given."""
    # NOTE(MthwRobinson) - changing the file_loader_cls to type here currently
    # results in pydantic validation errors
    file_loader_cls: Any = None
    """The file loader class to use."""
    file_loader_kwargs: Dict["str", Any] = {}
    """The file loader kwargs to use."""

    @root_validator
    def validate_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either folder_id or document_ids is set, but not both."""
        if values.get("folder_id") and (
            values.get("document_ids") or values.get("file_ids")
        ):
            raise ValueError(
                "Cannot specify both folder_id and document_ids nor "
                "folder_id and file_ids"
            )
        if (
            not values.get("folder_id")
            and not values.get("document_ids")
            and not values.get("file_ids")
        ):
            raise ValueError("Must specify either folder_id, document_ids, or file_ids")

        file_types = values.get("file_types")
        if file_types:
            if values.get("document_ids") or values.get("file_ids"):
                raise ValueError(
                    "file_types can only be given when folder_id is given,"
                    " (not when document_ids or file_ids are given)."
                )
            type_mapping = {
                "document": "application/vnd.google-apps.document",
                "sheet": "application/vnd.google-apps.spreadsheet",
                "pdf": "application/pdf",
            }
            allowed_types = list(type_mapping.keys()) + list(type_mapping.values())
            short_names = ", ".join([f"'{x}'" for x in type_mapping.keys()])
            full_names = ", ".join([f"'{x}'" for x in type_mapping.values()])
            for file_type in file_types:
                if file_type not in allowed_types:
                    raise ValueError(
                        f"Given file type {file_type} is not supported. "
                        f"Supported values are: {short_names}; and "
                        f"their full-form names: {full_names}"
                    )

            # replace short-form file types by full-form file types
            def full_form(x: str) -> str:
                return type_mapping[x] if x in type_mapping else x

            values["file_types"] = [full_form(file_type) for file_type in file_types]
        return values

    @validator("credentials_path")
    def validate_credentials_path(cls, v: Any, **kwargs: Any) -> Any:
        """Validate that credentials_path exists."""
        if not v.exists():
            raise ValueError(f"credentials_path {v} does not exist")
        return v

    def _load_credentials(self) -> Any:
        """Load credentials."""
        # Adapted from https://developers.google.com/drive/api/v3/quickstart/python
        try:
            from google.auth import default
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "google-api-python-client google-auth-httplib2 "
                "google-auth-oauthlib` "
                "to use the Google Drive loader."
            )

        creds = None
        if self.service_account_key.exists():
            return service_account.Credentials.from_service_account_file(
                str(self.service_account_key), scopes=SCOPES
            )

        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
                creds, project = default()
                creds = creds.with_scopes(SCOPES)
                # no need to write to file
                if creds:
                    return creds
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    def _load_sheet_from_id(self, id: str) -> List[Document]:
        """Load a sheet and all tabs from an ID."""

        from googleapiclient.discovery import build

        creds = self._load_credentials()
        sheets_service = build("sheets", "v4", credentials=creds)
        spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=id).execute()
        sheets = spreadsheet.get("sheets", [])

        documents = []
        for sheet in sheets:
            sheet_name = sheet["properties"]["title"]
            result = (
                sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=id, range=sheet_name)
                .execute()
            )
            values = result.get("values", [])
            if not values:
                continue  # empty sheet

            header = values[0]
            for i, row in enumerate(values[1:], start=1):
                metadata = {
                    "source": (
                        f"https://docs.google.com/spreadsheets/d/{id}/"
                        f"edit?gid={sheet['properties']['sheetId']}"
                    ),
                    "title": f"{spreadsheet['properties']['title']} - {sheet_name}",
                    "row": i,
                }
                content = []
                for j, v in enumerate(row):
                    title = header[j].strip() if len(header) > j else ""
                    content.append(f"{title}: {v.strip()}")

                page_content = "\n".join(content)
                documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def _load_document_from_id(self, id: str) -> Document:
        """Load a document from an ID."""
        from io import BytesIO

        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaIoBaseDownload

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)

        file = (
            service.files()
            .get(fileId=id, supportsAllDrives=True, fields="modifiedTime,name")
            .execute()
        )
        request = service.files().export_media(fileId=id, mimeType="text/plain")
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        try:
            while done is False:
                status, done = downloader.next_chunk()

        except HttpError as e:
            if e.resp.status == 404:
                print("File not found: {}".format(id))  # noqa: T201
            else:
                print("An error occurred: {}".format(e))  # noqa: T201

        text = fh.getvalue().decode("utf-8")
        metadata = {
            "source": f"https://docs.google.com/document/d/{id}/edit",
            "title": f"{file.get('name')}",
            "when": f"{file.get('modifiedTime')}",
        }
        return Document(page_content=text, metadata=metadata)

    def _load_documents_from_folder(
        self, folder_id: str, *, file_types: Optional[Sequence[str]] = None
    ) -> List[Document]:
        """Load documents from a folder."""
        from googleapiclient.discovery import build

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)
        files = self._fetch_files_recursive(service, folder_id)
        # If file types filter is provided, we'll filter by the file type.
        if file_types:
            _files = [f for f in files if f["mimeType"] in file_types]  # type: ignore
        else:
            _files = files

        returns = []
        for file in _files:
            if file["trashed"] and not self.load_trashed_files:
                continue
            elif file["mimeType"] == "application/vnd.google-apps.document":
                returns.append(self._load_document_from_id(file["id"]))  # type: ignore
            elif file["mimeType"] == "application/vnd.google-apps.spreadsheet":
                returns.extend(self._load_sheet_from_id(file["id"]))  # type: ignore
            elif (
                file["mimeType"] == "application/pdf"
                or self.file_loader_cls is not None
            ):
                returns.extend(self._load_file_from_id(file["id"]))  # type: ignore
            else:
                pass
        return returns

    def _fetch_files_recursive(
        self, service: Any, folder_id: str
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """Fetch all files and subfolders recursively."""
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents",
                pageSize=1000,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields="nextPageToken, files(id, name, mimeType, parents, trashed)",
            )
            .execute()
        )
        files = results.get("files", [])
        returns = []
        for file in files:
            if file["mimeType"] == "application/vnd.google-apps.folder":
                if self.recursive:
                    returns.extend(self._fetch_files_recursive(service, file["id"]))
            else:
                returns.append(file)

        return returns

    def _load_documents_from_ids(self) -> List[Document]:
        """Load documents from a list of IDs."""
        if not self.document_ids:
            raise ValueError("document_ids must be set")

        return [self._load_document_from_id(doc_id) for doc_id in self.document_ids]

    def _load_file_from_id(self, id: str) -> List[Document]:
        """Load a file from an ID."""
        from io import BytesIO

        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)

        file = service.files().get(fileId=id, supportsAllDrives=True).execute()
        request = service.files().get_media(fileId=id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        if self.file_loader_cls is not None:
            fh.seek(0)
            loader = self.file_loader_cls(file=fh, **self.file_loader_kwargs)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = f"https://drive.google.com/file/d/{id}/view"
                if "title" not in doc.metadata:
                    doc.metadata["title"] = f"{file.get('name')}"
            return docs

        else:
            from PyPDF2 import PdfReader

            content = fh.getvalue()
            pdf_reader = PdfReader(BytesIO(content))

            return [
                Document(
                    page_content=page.extract_text(),
                    metadata={
                        "source": f"https://drive.google.com/file/d/{id}/view",
                        "title": f"{file.get('name')}",
                        "page": i,
                    },
                )
                for i, page in enumerate(pdf_reader.pages)
            ]

    def _load_file_from_ids(self) -> List[Document]:
        """Load files from a list of IDs."""
        if not self.file_ids:
            raise ValueError("file_ids must be set")
        docs = []
        for file_id in self.file_ids:
            docs.extend(self._load_file_from_id(file_id))
        return docs

    def load(self) -> List[Document]:
        """Load documents."""
        if self.folder_id:
            return self._load_documents_from_folder(
                self.folder_id, file_types=self.file_types
            )
        elif self.document_ids:
            return self._load_documents_from_ids()
        else:
            return self._load_file_from_ids()
