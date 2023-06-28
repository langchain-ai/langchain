"""Loading logic for loading documents from an GCS directory."""
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.gcs_file import GCSFileLoader


class GCSDirectoryLoader(BaseLoader):
    """Loads Documents from GCS."""

    def __init__(self, project_name: str, bucket: str, prefix: str = ""):
        """Initialize with bucket and key name.

        Args:
            project_name: The name of the project for the GCS bucket.
            bucket: The name of the GCS bucket.
            prefix: The prefix of the GCS bucket.
        """
        self.project_name = project_name
        self.bucket = bucket
        self.prefix = prefix

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "Could not import google-cloud-storage python package. "
                "Please install it with `pip install google-cloud-storage`."
            )
        client = storage.Client(project=self.project_name)
        docs = []
        for blob in client.list_blobs(self.bucket, prefix=self.prefix):
            # we shall just skip directories since GCSFileLoader creates
            # intermediate directories on the fly
            if blob.name.endswith("/"):
                continue
            loader = GCSFileLoader(self.project_name, self.bucket, blob.name)
            docs.extend(loader.load())
        return docs
