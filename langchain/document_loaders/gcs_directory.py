"""Loading logic for loading documents from an GCS directory."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.gcs_file import GCSFileLoader


class GCSDirectoryLoader(BaseLoader):
    """Loading logic for loading documents from GCS."""

    def __init__(self, project_name: str, bucket: str, prefix: str = ""):
        """Initialize with bucket and key name."""
        self.project_name = project_name
        self.bucket = bucket
        self.prefix = prefix

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ValueError(
                "Could not import google-cloud-storage python package. "
                "Please it install it with `pip install google-cloud-storage`."
            )
        client = storage.Client(project=self.project_name)
        docs = []
        for blob in client.list_blobs(self.bucket, prefix=self.prefix):
            loader = GCSFileLoader(self.project_name, self.bucket, blob.name)
            docs.extend(loader.load())
        return docs
