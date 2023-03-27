"""Loading logic for loading documents from an Azure Blob Storage container."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.azure_blob_storage_file import (
    AzureBlobStorageFileLoader,
)
from langchain.document_loaders.base import BaseLoader


class AzureBlobStorageContainerLoader(BaseLoader):
    """Loading logic for loading documents from Azure Blob Storage."""

    def __init__(self, conn_str: str, container: str, prefix: str = ""):
        """Initialize with connection string, container and blob prefix."""
        self.conn_str = conn_str
        self.container = container
        self.prefix = prefix

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:
            raise ValueError(
                "Could not import azure storage blob python package. "
                "Please it install it with `pip install azure-storage-blob`."
            ) from exc

        container = ContainerClient.from_connection_string(
            conn_str=self.conn_str, container_name=self.container
        )
        docs = []
        blob_list = container.list_blobs(name_starts_with=self.prefix)
        for blob in blob_list:
            loader = AzureBlobStorageFileLoader(
                self.conn_str, self.container, blob.name  # type: ignore
            )
            docs.extend(loader.load())
        return docs
