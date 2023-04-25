"""Loader that loads HuggingFace datasets."""
import itertools
from typing import (
    Any,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

T = TypeVar("T")


def get_n_elements(iterator: Iterator[T], n: int) -> List[T]:
    return list(itertools.islice(iterator, n))


class HuggingFaceDatasetLoader(BaseLoader):
    """Loading logic for loading documents from the Hugging Face Hub."""

    def __init__(
        self,
        path: str,
        page_content_column: str = "text",
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[
            Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
        ] = None,
        cache_dir: Optional[str] = None,
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        num_proc: Optional[int] = None,
        streaming: bool = False,
        batch_size: int = 10,
    ):
        """
        Initialize the HuggingFaceDatasetLoader.
        Args:
            path: Path or name of the dataset.
            page_content_column: Page content column name.
            name: Name of the dataset configuration.
            data_dir: Data directory of the dataset configuration.
            data_files: Path(s) to source data file(s).
            cache_dir: Directory to read/write data.
            keep_in_memory: Whether to copy the dataset in-memory.
            save_infos: Save the dataset information (checksums/size/splits/...).
            use_auth_token: Bearer token for remote files on the Datasets Hub.
            num_proc: Number of processes.
            streaming:streams the data progressively while iterating on the dataset
            batch_size: batch_size for streaming dataset
        """
        self.path = path
        self.page_content_column = page_content_column
        self.name = name
        self.data_dir = data_dir
        self.data_files = data_files
        self.cache_dir = cache_dir
        self.keep_in_memory = keep_in_memory
        self.save_infos = save_infos
        self.use_auth_token = use_auth_token
        self.num_proc = num_proc
        self.streaming = streaming
        self.batch_size = batch_size
        self._iterator: Optional[Any] = None

    def load(self) -> List[Document]:
        if self._iterator is not None:
            docs = [
                Document(
                    page_content=row.pop(self.page_content_column),
                    metadata=row,
                )
                for row in get_n_elements(self._iterator, self.batch_size)
            ]
            return docs

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Could not import datasets python package. "
                "Please install it with `pip install datasets`."
            )

        dataset = load_dataset(
            path=self.path,
            name=self.name,
            data_dir=self.data_dir,
            data_files=self.data_files,
            cache_dir=self.cache_dir,
            keep_in_memory=self.keep_in_memory,
            save_infos=self.save_infos,
            use_auth_token=self.use_auth_token,
            num_proc=self.num_proc,
            streaming=self.streaming,
        )

        if self.streaming:
            iterators = [iter(dataset[split]) for split in dataset.keys()]
            self._iterator = itertools.chain(*iterators)

            docs = [
                Document(
                    page_content=row.pop(self.page_content_column),
                    metadata=row,
                )
                for row in get_n_elements(self._iterator, self.batch_size)
            ]
            return docs

        docs = [
            Document(
                page_content=row.pop(self.page_content_column),
                metadata=row,
            )
            for key in dataset.keys()
            for row in dataset[key]
        ]

        return docs
