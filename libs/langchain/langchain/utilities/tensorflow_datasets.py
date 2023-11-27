import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)


class TensorflowDatasets(BaseModel):
    """Access to the TensorFlow Datasets.

    The Current implementation can work only with datasets that fit in a memory.

    `TensorFlow Datasets` is a collection of datasets ready to use, with TensorFlow
    or other Python ML frameworks, such as Jax. All datasets are exposed
    as `tf.data.Datasets`.
    To get started see the Guide: https://www.tensorflow.org/datasets/overview and
    the list of datasets: https://www.tensorflow.org/datasets/catalog/
                                               overview#all_datasets

    You have to provide the sample_to_document_function: a function that
       a sample from the dataset-specific format to the Document.

    Attributes:
        dataset_name: the name of the dataset to load
        split_name: the name of the split to load. Defaults to "train".
        load_max_docs: a limit to the number of loaded documents. Defaults to 100.
        sample_to_document_function: a function that converts a dataset sample
          to a Document

    Example:
        .. code-block:: python

            from langchain.utilities import TensorflowDatasets

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

            tsds_client = TensorflowDatasets(
                    dataset_name="mlqa/en",
                    split_name="train",
                    load_max_docs=MAX_DOCS,
                    sample_to_document_function=mlqaen_example_to_document,
                )

    """

    dataset_name: str = ""
    split_name: str = "train"
    load_max_docs: int = 100
    sample_to_document_function: Optional[Callable[[Dict], Document]] = None
    dataset: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import tensorflow python package. "
                "Please install it with `pip install tensorflow`."
            )
        try:
            import tensorflow_datasets
        except ImportError:
            raise ImportError(
                "Could not import tensorflow_datasets python package. "
                "Please install it with `pip install tensorflow-datasets`."
            )
        if values["sample_to_document_function"] is None:
            raise ValueError(
                "sample_to_document_function is None. "
                "Please provide a function that converts a dataset sample to"
                "  a Document."
            )
        values["dataset"] = tensorflow_datasets.load(
            values["dataset_name"], split=values["split_name"]
        )

        return values

    def lazy_load(self) -> Iterator[Document]:
        """Download a selected dataset lazily.

        Returns: an iterator of Documents.

        """
        return (
            self.sample_to_document_function(s)
            for s in self.dataset.take(self.load_max_docs)
            if self.sample_to_document_function is not None
        )

    def load(self) -> List[Document]:
        """Download a selected dataset.

        Returns: a list of Documents.

        """
        return list(self.lazy_load())
