"""Loading logic for loading documents from an AWS S3 file."""
import os
import tempfile
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class S3FileLoader(BaseLoader):
    """Loading logic for loading documents from an AWS S3 file."""

    def __init__(self, bucket: str, key: str):
        """Initialize with bucket and key name.

        Args:
            bucket: The name of the S3 bucket.
            key: The key of the S3 object.
        """
        self.bucket = bucket
        self.key = key

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import `boto3` python package. "
                "Please install it with `pip install boto3`."
            )
        s3 = boto3.client("s3")
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3.download_file(self.bucket, self.key, file_path)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
