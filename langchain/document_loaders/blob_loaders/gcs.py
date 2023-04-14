"""Loading logic for loading documents from an GCS directory."""
from io import BytesIO
from typing import Generator

from langchain.document_loaders.blob_loaders.schema import Blob, BlobLoader


class GCSBlobLoader(BlobLoader):
    """GSC blob loader."""

    def __init__(self, project_name: str, bucket: str, prefix: str = "") -> None:
        """Initialize with bucket and key name."""
        self.project_name = project_name
        self.bucket = bucket
        self.prefix = prefix

    def yield_blobs(
        self,
    ) -> Generator[Blob, None, None]:
        """Yield blobs matching the given pattern."""

        try:
            from google.cloud import storage
        except ImportError:
            raise ValueError(
                "Could not import google-cloud-storage python package. "
                "Please install it with `pip install google-cloud-storage`."
            )
        client = storage.Client(project=self.project_name)
        bytes_io = BytesIO()
        for blob in client.list_blobs(self.bucket, prefix=self.prefix):
            client.download_blob_to_file(blob, bytes_io)
            yield Blob(path_like=blob.name, data=bytes_io.seek(0))
