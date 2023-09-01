import os
import tempfile
from typing import List

from langchain.document_loaders.unstructured import UnstructuredBaseLoader


class S3FileLoader(UnstructuredBaseLoader):
    """Load from `Amazon AWS S3` file."""

    def __init__(self, bucket: str, key: str):
        """Initialize with bucket and key name.

        Args:
            bucket: The name of the S3 bucket.
            key: The key of the S3 object.
        """
        super().__init__()
        self.bucket = bucket
        self.key = key

    def _get_elements(self) -> List:
        """Get elements."""
        from unstructured.partition.auto import partition

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
            return partition(filename=file_path)

    def _get_metadata(self) -> dict:
        return {"source": f"s3://{self.bucket}/{self.key}"}
