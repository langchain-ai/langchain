from typing import Callable, Dict, Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.tensorflow_datasets import TensorflowDatasets


class TensorflowDatasetLoader(BaseLoader):
    """Load from `TensorFlow Dataset`.

    Attributes:
        dataset_name: the name of the dataset to load
        split_name: the name of the split to load.
        load_max_docs: a limit to the number of loaded documents. Defaults to 100.
        sample_to_document_function: a function that converts a dataset sample
          into a Document

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import TensorflowDatasetLoader

            def mlqaen_example_to_document(example: dict) -> Document:
                return Document(
                    page_content=decode_to_str(example["context"]),
                    metadata={
                        "id": decode_to_str(example["id"]),
                        "title": decode_to_str(example["title"]),
                        "question": decode_to_str(example["question"]),
                        "answer": decode_to_str(example["answers"]["text"][0]),
                    },
                )

            tsds_client = TensorflowDatasetLoader(
                    dataset_name="mlqa/en",
                    split_name="test",
                    load_max_docs=100,
                    sample_to_document_function=mlqaen_example_to_document,
                )

    """

    def __init__(
        self,
        dataset_name: str,
        split_name: str,
        load_max_docs: Optional[int] = 100,
        sample_to_document_function: Optional[Callable[[Dict], Document]] = None,
    ):
        """Initialize the TensorflowDatasetLoader.

        Args:
            dataset_name: the name of the dataset to load
            split_name: the name of the split to load.
            load_max_docs: a limit to the number of loaded documents. Defaults to 100.
            sample_to_document_function: a function that converts a dataset sample
                into a Document.
        """
        self.dataset_name: str = dataset_name
        self.split_name: str = split_name
        self.load_max_docs = load_max_docs
        """The maximum number of documents to load."""
        self.sample_to_document_function: Optional[
            Callable[[Dict], Document]
        ] = sample_to_document_function
        """Custom function that transform a dataset sample into a Document."""

        self._tfds_client = TensorflowDatasets(
            dataset_name=self.dataset_name,
            split_name=self.split_name,
            load_max_docs=self.load_max_docs,
            sample_to_document_function=self.sample_to_document_function,
        )

    def lazy_load(self) -> Iterator[Document]:
        yield from self._tfds_client.lazy_load()
