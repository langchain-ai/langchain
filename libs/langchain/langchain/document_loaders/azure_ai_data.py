import os
import tempfile
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class AzureAIDataLoader(BaseLoader):
    """Load from Azure AI Data."""

    def __init__(self, url: str, glob_pattern: Optional[str] = None):
        """Initialize with URL to a data asset or storage location
        ."""
        self.url = url
        """URL to the data asset or storage location."""
        self.glob_pattern = glob_pattern
        """Optional glob pattern to select files. Defaults to None."""

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from azureml.fsspec import AzureMachineLearningFileSystem
        except ImportError as exc:
            raise ImportError(
                "Could not import azureml-fspec package."
                "Please install it with `pip install azureml-fsspec`."
            ) from exc

        fs = AzureMachineLearningFileSystem(self.url)

        remote_paths_list = []
        if self.glob_pattern:
            remote_paths_list = fs.glob(self.glob_pattern)
        else:
            remote_paths_list = fs.ls()

        docs = []
        for remote_path in remote_paths_list:
            with tempfile.TemporaryDirectory() as tmp:
                local_path = os.path.join(tmp, os.path.basename(remote_path))
                fs.download(remote_path, tmp)
                # check file exists
                if os.path.exists(local_path):
                    loader = UnstructuredFileLoader(local_path)
                    docs.extend(loader.load())

        return docs
