"""Loading logic for loading documents from an GCS directory."""
from typing import Union, Generator, List
from io import BytesIO

from langchain.docstore.document import Document
from langchain.document_loaders.base import Blob, BaseLoader


class GCSDirectoryLoader(BaseLoader):
    """Loading logic for loading documents from GCS."""

    def __init__(self, project_name: str, bucket: str, prefix: str = "") -> None:
        """Initialize with bucket and key name."""
        self.project_name = project_name
        self.bucket = bucket
        self.prefix = prefix

    def load(self) -> List[Document]:
        raise AssertionError("Do not use.")

    def lazy_load(
        self,
    ) -> Union[Generator[Blob, None, None], Generator[Document, None, None]]:
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
