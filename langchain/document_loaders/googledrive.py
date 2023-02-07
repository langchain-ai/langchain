"""Loader that loads data from Google Drive."""

# Prerequisites:
# 1. Create a Google Cloud project
# 2. Enable the Google Drive API:
#   https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com
# 3. Authorize credentials for desktop app:
#   https://developers.google.com/drive/api/quickstart/python#authorize_credentials_for_a_desktop_application # noqa: E501


from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator, validator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveLoader(BaseLoader, BaseModel):
    """Loader that loads Google Docs from Google Drive."""

    credentials_path: Path = Path.home() / ".credentials" / "credentials.json"
    token_path: Path = Path.home() / ".credentials" / "token.json"
    folder_id: Optional[str] = None
    document_ids: Optional[List[str]] = None

    @root_validator
    def validate_folder_id_or_document_ids(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that either folder_id or document_ids is set, but not both."""
        if values.get("folder_id") and values.get("document_ids"):
            raise ValueError("Cannot specify both folder_id and document_ids")
        if not values.get("folder_id") and not values.get("document_ids"):
            raise ValueError("Must specify either folder_id or document_ids")
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
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            raise ImportError(
                "You must run"
                "`pip install --upgrade "
                "google-api-python-client google-auth-httplib2 "
                "google-auth-oauthlib`"
                "to use the Google Drive loader."
            )

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    def _load_document_from_id(self, id: str) -> Document:
        """Load a document from an ID."""
        from io import BytesIO

        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)

        request = service.files().export_media(fileId=id, mimeType="text/plain")
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        text = fh.getvalue().decode("utf-8")
        metadata = {"source": f"https://docs.google.com/document/d/{id}/edit"}
        return Document(page_content=text, metadata=metadata)

    def _load_documents_from_folder(self) -> List[Document]:
        """Load documents from a folder."""
        from googleapiclient.discovery import build

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)

        results = (
            service.files()
            .list(
                q=f"'{self.folder_id}' in parents",
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType)",
            )
            .execute()
        )
        items = results.get("files", [])

        docs = []
        for item in items:
            # Only support Google Docs for now
            if item["mimeType"] == "application/vnd.google-apps.document":
                docs.append(self._load_document_from_id(item["id"]))
        return docs

    def _load_documents_from_ids(self) -> List[Document]:
        """Load documents from a list of IDs."""
        if not self.document_ids:
            raise ValueError("document_ids must be set")

        docs = []
        for doc_id in self.document_ids:
            docs.append(self._load_document_from_id(doc_id))
        return docs

    def load(self) -> List[Document]:
        """Load documents."""
        if self.folder_id:
            return self._load_documents_from_folder()
        else:
            return self._load_documents_from_ids()
