from typing import Iterator, Optional

from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileIOLoader


class AzureAIDataLoader(BaseLoader):
    """Load from Azure AI Data."""

    def __init__(self, url: str, glob: Optional[str] = None):
        """Initialize with URL to a data asset or storage location
        ."""
        self.url = url
        """URL to the data asset or storage location."""
        self.glob_pattern = glob
        """Optional glob pattern to select files. Defaults to None."""

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        try:
            from azureml.fsspec import AzureMachineLearningFileSystem
        except ImportError as exc:
            raise ImportError(
                "Could not import azureml-fspec package."
                "Please install it with `pip install azureml-fsspec`."
            ) from exc

        fs = AzureMachineLearningFileSystem(self.url)

        if self.glob_pattern:
            remote_paths_list = fs.glob(self.glob_pattern)
        else:
            remote_paths_list = fs.ls()

        for remote_path in remote_paths_list:
            with fs.open(remote_path) as f:
                loader = UnstructuredFileIOLoader(file=f)
                yield from loader.load()