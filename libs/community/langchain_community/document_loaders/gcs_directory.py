import logging
from typing import Callable, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.gcs_file import GCSFileLoader
from langchain_community.utilities.vertexai import get_client_info

logger = logging.getLogger(__name__)


class GCSDirectoryLoader(BaseLoader):
    """Load from GCS directory."""

    def __init__(
        self,
        project_name: str,
        bucket: str,
        prefix: str = "",
        loader_func: Optional[Callable[[str], BaseLoader]] = None,
        continue_on_failure: bool = False,
    ):
        """Initialize with bucket and key name.

        Args:
            project_name: The ID of the project for the GCS bucket.
            bucket: The name of the GCS bucket.
            prefix: The prefix of the GCS bucket.
            loader_func: A loader function that instantiates a loader based on a
                file_path argument. If nothing is provided, the  GCSFileLoader
                would use its default loader.
            continue_on_failure: To use try-except block for each file within the GCS
                directory. If set to `True`, then failure to process a file will not
                cause an error.
        """
        self.project_name = project_name
        self.bucket = bucket
        self.prefix = prefix
        self._loader_func = loader_func
        self.continue_on_failure = continue_on_failure

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "Could not import google-cloud-storage python package. "
                "Please install it with `pip install google-cloud-storage`."
            )
        client = storage.Client(
            project=self.project_name,
            client_info=get_client_info(module="google-cloud-storage"),
        )
        docs = []
        for blob in client.list_blobs(self.bucket, prefix=self.prefix):
            # we shall just skip directories since GCSFileLoader creates
            # intermediate directories on the fly
            if blob.name.endswith("/"):
                continue
            loader = GCSFileLoader(
                self.project_name, self.bucket, blob.name, loader_func=self._loader_func
            )
            docs.extend(loader.load())
            # Use the try-except block here
            try:
                loader = GCSFileLoader(
                    self.project_name,
                    self.bucket,
                    blob.name,
                    loader_func=self._loader_func,
                )
                docs.extend(loader.load())
            except Exception as e:
                if self.continue_on_failure:
                    logger.warning(f"Problem processing blob {blob.name}, message: {e}")
                    continue
                else:
                    raise e
        return docs
