"""Load documents from a GCS file."""
import os
import tempfile
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class GCSFileLoader(BaseLoader):
    """Load Documents from a GCS file."""

    def __init__(self, project_name: str, bucket: str, blob: str):
        """Initialize with bucket and key name.

        Args:
            project_name: The name of the project to load
            bucket: The name of the GCS bucket.
            blob: The name of the GCS blob to load.
        """
        self.bucket = bucket
        self.blob = blob
        self.project_name = project_name

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "Could not import google-cloud-storage python package. "
                "Please install it with `pip install google-cloud-storage`."
            )

        # Initialise a client
        storage_client = storage.Client(self.project_name)
        # Create a bucket object for our bucket
        bucket = storage_client.get_bucket(self.bucket)
        # Create a blob object from the filepath
        blob = bucket.blob(self.blob)
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.blob}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Download the file to a destination
            blob.download_to_filename(file_path)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
