from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders.azure_blob_storage_file import (
    AzureBlobStorageFileLoader,
)
from langchain_community.document_loaders.base import BaseLoader


class AzureBlobStorageContainerLoader(BaseLoader):
    """Load from `Azure Blob Storage` container."""

    def __init__(self, conn_str: str, container: str, prefix: str = ""):
        """Initialize with connection string, container and blob prefix."""
        self.conn_str = conn_str
        """Connection string for Azure Blob Storage."""
        self.container = container
        """Container name."""
        self.prefix = prefix
        """Prefix for blob names."""

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:
            raise ImportError(
                "Could not import azure storage blob python package. "
                "Please install it with `pip install azure-storage-blob`."
            ) from exc

        container = ContainerClient.from_connection_string(
            conn_str=self.conn_str, container_name=self.container
        )
        docs = []
        blob_list = container.list_blobs(name_starts_with=self.prefix)
        for blob in blob_list:
            loader = AzureBlobStorageFileLoader(
                self.conn_str,
                self.container,
                blob.name,
            )
            docs.extend(loader.load())
        return docs