from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class KaggleDatasetLoader(BaseLoader):
    """Load from `Kaggle` datasets.

    Follow these steps to use this loader:
    - Register a Kaggle account and create an API token to use this loader.
    See https://www.kaggle.com/settings
    - Install `kaggle` python package with `pip install kaggle`
    - Use `kaggle datasets list` to list all available datasets
    - Use `kaggle datasets <dataset_name>` to download the dataset
    - Use `unzip <dataset_zipfile_name>` to extract all files in the dataset
    - Open the dataset CSV file and choose the column name for page content
    - Use the dataset CSV file name and the column name to
      initialize the KaggleDatasetLoader

    Note: Other columns in the dataset CSV file will be treated as metadata.
    """

    def __init__(self, dataset_path: str, page_content_column: str):
        """Initialize the KaggleDatasetLoader.

        Args:
            dataset_path: Path to the dataset CSV file.
            page_content_column: Page content column name.
        """

        self.dataset_path = dataset_path
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily."""
        for doc in self.load():
            yield doc

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Could not import pandas python package. "
                "Please install it with `pip install pandas`."
            )

        df = pd.read_csv(self.dataset_path)
        docs = []
        df.apply(lambda row: docs.append(self._sample2document(row)), axis=1)
        return docs

    def _sample2document(self, sample: Any) -> Document:
        """Convert a pandas dataframe sample into a Document

        The `content_field` goes into the document content,
        all other fields go into the metadata.
        """
        assert self.page_content_column in sample.index, (
            f"content field {self.page_content_column} "
            f"should be in the sample columns: {sample}"
        )
        return Document(
            page_content=str(sample[self.page_content_column])
            if sample[self.page_content_column] is not None
            else "",
            metadata={
                k: v
                for k, v in sample.to_dict().items()
                if k != self.page_content_column
            },
        )
