import os
import tempfile
from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class AzureBlobStorageFileLoader(BaseLoader):
    """Load from `Azure Blob Storage` files."""

    def __init__(self, conn_str: str, container: str, blob_name: str):
        """Initialize with connection string, container and blob name."""
        self.conn_str = conn_str
        """Connection string for Azure Blob Storage."""
        self.container = container
        """Container name."""
        self.blob = blob_name
        """Blob name."""

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from azure.storage.blob import BlobClient
        except ImportError as exc:
            raise ImportError(
                "Could not import azure storage blob python package. "
                "Please install it with `pip install azure-storage-blob`."
            ) from exc

        client = BlobClient.from_connection_string(
            conn_str=self.conn_str, container_name=self.container, blob_name=self.blob
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.container}/{self.blob}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(f"{file_path}", "wb") as file:
                blob_data = client.download_blob()
                blob_data.readinto(file)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
